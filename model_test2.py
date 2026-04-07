from pathlib import Path
import torch,time
from transformers import AutoModelForCausalLM, AutoTokenizer


# Set model path directly (current project path)
MODEL_PATH = Path("models/qwen3.5-0.8b-chatdoctor-unsloth/checkpoint-700")
#MODEL_PATH = Path("models/base/Qwen3.5-0.8B")

# SYSTEM_PROMPT = (
#     "You are a cautious medical question answering assistant. "
#     "Provide general educational guidance, avoid definitive diagnosis, "
#     "and recommend professional care when appropriate."
# )

SYSTEM_PROMPT = (
    "You are a cautious medical question answering assistant. "
    "Answer briefly and clearly in maximum 8-10 lines. "
    "Give only essential advice, no long explanations. "
    "Avoid diagnosis and suggest consulting a doctor if needed.")

def load_model(model_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto",  # important
        trust_remote_code=True,
    )

    model.eval()
    print("Model device:", next(model.parameters()).device)
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
def generate_answer(model, tokenizer, device, question: str, max_new_tokens: int = 240) -> str:
    prompt = build_prompt(tokenizer, question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

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


def main():
    # Write your question here directly
    question1 = "What should I do for a mild fever ?"
    question2 = "What should I do for a Headache ?"

    model, tokenizer, device = load_model(MODEL_PATH)
    t1= time.time()
    answer = generate_answer(model, tokenizer, device, question1)
    print(f'time taken by model: {round(time.time()-t1,2)}')
    print(" Question:")
    print(question1)

    print("Answer:")
    print(answer)


    t1= time.time()
    answer = generate_answer(model, tokenizer, device, question2)
    print(f'time taken by model: {round(time.time()-t1,2)}')
    print("Question:")
    print(question2)

    print("Answer:")
    print(answer)



if __name__ == "__main__":
    main()