import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('UNSLOTH_FUSED_CE_COMPILE_DISABLE', '1')
os.environ.setdefault('UNSLOTH_CE_LOSS_TARGET_GB', '0.05')

import matplotlib.pyplot as plt
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer

DEFAULT_MODEL_PATH = 'models/base/Qwen3.5-0.8B'
DEFAULT_OUTPUT_DIR = 'models/qwen3.5-0.8b-chatdoctor-unsloth'
SYSTEM_PROMPT = (
    'You are a cautious medical question answering assistant. '
    'Provide general educational guidance, avoid definitive diagnosis, '
    'and recommend professional care when appropriate.'
)
TARGET_MODULES = [
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
]


def format_example(example, tokenizer):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': example['instruction'].strip()},
        {'role': 'assistant', 'content': example['response'].strip()},
    ]
    return {
        'text': tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def plot_losses(log_history, output_dir: Path):
    train_points = [(item['step'], item['loss']) for item in log_history if 'loss' in item and 'eval_loss' not in item]
    eval_points = [(item['step'], item['eval_loss']) for item in log_history if 'eval_loss' in item]
    if not train_points and not eval_points:
        return

    plt.figure(figsize=(8, 5))
    if train_points:
        xs, ys = zip(*train_points)
        plt.plot(xs, ys, label='train_loss')
    if eval_points:
        xs, ys = zip(*eval_points)
        plt.plot(xs, ys, label='validation_loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=160)
    plt.close()


def save_outputs(trainer, tokenizer, output_dir: Path, save_merged: bool) -> dict:
    adapter_dir = output_dir / 'adapter'
    adapter_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    saved_paths = {
        'adapter_dir': str(adapter_dir),
        'merged_model_dir': None,
    }

    if save_merged:
        if not hasattr(trainer.model, 'save_pretrained_merged'):
            raise AttributeError('This Unsloth build does not expose save_pretrained_merged.')
        trainer.model.save_pretrained_merged(
            str(output_dir),
            tokenizer,
            save_method='merged_16bit',
        )
        saved_paths['merged_model_dir'] = str(output_dir)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Fine-tune ChatDoctor medical QA with Unsloth QLoRA.')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--validation_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--torch_empty_cache_steps', type=int, default=10)
    parser.add_argument(
        '--save_merged',
        action='store_true',
        help='Also save a merged 16-bit fine-tuned model in output_dir in addition to the LoRA adapter.',
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_name)
    if not model_path.exists():
        raise FileNotFoundError(f'Base model path not found: {model_path}')

    dataset = load_dataset(
        'json',
        data_files={
            'train': args.train_file,
            'validation': args.validation_file,
        },
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing='unsloth',
        random_state=42,
    )

    formatted = dataset.map(
        lambda row: format_example(row, tokenizer),
        remove_columns=dataset['train'].column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        logging_steps=args.logging_steps,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',
        fp16=False,
        bf16=True,
        optim='adamw_8bit',
        torch_empty_cache_steps=args.torch_empty_cache_steps,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted['train'],
        eval_dataset=formatted['validation'],
        dataset_text_field='text',
        max_seq_length=args.max_seq_length,
        packing=False,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    saved_paths = save_outputs(trainer, tokenizer, output_dir, save_merged=args.save_merged)

    log_history = trainer.state.log_history
    with (output_dir / 'training_log.json').open('w', encoding='utf-8') as handle:
        json.dump(log_history, handle, indent=2)

    plot_losses(log_history, output_dir)

    summary = {
        'model_name': args.model_name,
        'framework': 'unsloth',
        'method': 'qlora',
        'dataset': 'chatdoctor',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'max_seq_length': args.max_seq_length,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'logging_steps': args.logging_steps,
        'eval_steps': args.eval_steps,
        'save_steps': args.save_steps,
        'torch_empty_cache_steps': args.torch_empty_cache_steps,
        'target_modules': TARGET_MODULES,
        'save_merged': args.save_merged,
        'saved_paths': saved_paths,
    }
    with (output_dir / 'run_config.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
