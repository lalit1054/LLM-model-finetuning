from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = Path("models/qwen3.5-0.8b-chatdoctor-unsloth")
DEFAULT_QUESTION = "What should I do for a mild fever and sore throat?"
SYSTEM_PROMPT = (
    "You are a cautious medical question answering assistant. "
    "Provide general educational guidance, avoid definitive diagnosis, "
    "and recommend professional care when appropriate."
)


def load_model(model_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"SYSTEM: {SYSTEM_PROMPT}\nUSER: {question.strip()}\nASSISTANT:"


@torch.inference_mode()
def generate_answer(model, tokenizer, device, question: str, max_new_tokens: int) -> str:
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_length = inputs["input_ids"].shape[1]
    answer_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick test on the fine-tuned model.")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the fine-tuned model directory.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question to ask the model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path)
    answer = generate_answer(model, tokenizer, device, args.question, args.max_new_tokens)
    print(answer)


if __name__ == "__main__":
    main()
