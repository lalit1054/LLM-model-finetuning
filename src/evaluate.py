import argparse
import json
import re
from collections import Counter
from pathlib import Path

from unsloth import FastLanguageModel
import evaluate
from datasets import load_dataset

DEFAULT_BASE_MODEL = 'models/base/Qwen3.5-0.8B'
DEFAULT_ADAPTER_PATH = 'models/qwen3.5-0.8b-chatdoctor-unsloth'
SYSTEM_PROMPT = (
    'You are a cautious medical question answering assistant. '
    'Provide general educational guidance, avoid definitive diagnosis, '
    'and recommend professional care when appropriate.'
)


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize_answer(text: str):
    return re.findall(r'\w+', text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction)
    ref_tokens = normalize_answer(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_unsloth_model(model_path: str, max_seq_length: int = 768):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 220) -> str:
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_metrics(predictions, references):
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    avg_f1 = sum(token_f1(pred, ref) for pred, ref in zip(predictions, references)) / max(len(predictions), 1)
    return {
        'rouge1': rouge_scores['rouge1'],
        'rougeL': rouge_scores['rougeL'],
        'bleu': bleu_score['bleu'],
        'token_f1': avg_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare base and fine-tuned ChatDoctor medical QA models.')
    parser.add_argument('--base_model', type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--adapter_path', type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--max_seq_length', type=int, default=768)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset('json', data_files={'test': args.test_file})['test']
    if args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))

    base_model, base_tokenizer = load_unsloth_model(args.base_model, args.max_seq_length)
    finetuned_model, finetuned_tokenizer = load_unsloth_model(args.adapter_path, args.max_seq_length)

    references = []
    base_predictions = []
    finetuned_predictions = []
    examples = []

    for row in dataset:
        question = row['instruction']
        reference = row['response']
        base_prediction = generate_answer(base_model, base_tokenizer, question)
        finetuned_prediction = generate_answer(finetuned_model, finetuned_tokenizer, question)

        references.append(reference)
        base_predictions.append(base_prediction)
        finetuned_predictions.append(finetuned_prediction)
        examples.append({
            'instruction': question,
            'reference': reference,
            'base_prediction': base_prediction,
            'finetuned_prediction': finetuned_prediction,
        })

    metrics = {
        'base_model': compute_metrics(base_predictions, references),
        'finetuned_model': compute_metrics(finetuned_predictions, references),
        'num_samples': len(references),
    }

    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    with (output_dir / 'examples.jsonl').open('w', encoding='utf-8') as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + '\n')

    report = [
        '# Evaluation Report',
        '',
        f'Samples evaluated: {len(references)}',
        '',
        '## Metrics',
        '',
        f"- Base ROUGE-1: {metrics['base_model']['rouge1']:.4f}",
        f"- Base ROUGE-L: {metrics['base_model']['rougeL']:.4f}",
        f"- Base BLEU: {metrics['base_model']['bleu']:.4f}",
        f"- Base Token F1: {metrics['base_model']['token_f1']:.4f}",
        f"- Fine-tuned ROUGE-1: {metrics['finetuned_model']['rouge1']:.4f}",
        f"- Fine-tuned ROUGE-L: {metrics['finetuned_model']['rougeL']:.4f}",
        f"- Fine-tuned BLEU: {metrics['finetuned_model']['bleu']:.4f}",
        f"- Fine-tuned Token F1: {metrics['finetuned_model']['token_f1']:.4f}",
        '',
        '## Qualitative Review Guidance',
        '',
        '- Select 5 to 10 rows from `examples.jsonl` for the final report.',
        '- Highlight cases where the fine-tuned model becomes more medically grounded or cautious.',
        '- Highlight failures involving dosage, escalation, or unsupported claims.',
        '- Discuss whether fine-tuning improved helpfulness at the cost of verbosity or overconfidence.',
    ]
    (output_dir / 'evaluation_report.md').write_text('\n'.join(report), encoding='utf-8')

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
