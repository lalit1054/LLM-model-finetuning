# %% [markdown]
# # Complete Evaluation and Analysis for QLoRA Medical QA
#
# This notebook-style script performs a professional evaluation of a QLoRA
# fine-tuned Qwen 0.8B model on the ChatDoctor test split.
#
# It covers:
# - Base model evaluation
# - Prompt-only baseline evaluation
# - Fine-tuned model evaluation
# - ROUGE, BLEU, and optional BERTScore
# - Metric visualizations
# - Loss curve analysis and overfitting check
# - Qualitative comparison
# - Heuristic error analysis
# - Saving predictions, metrics, plots, and a text report
#
# The code is organized into reusable functions and is designed to run on CPU
# with conservative memory usage.

# %% [markdown]
# ## 1. Imports and Configuration

# %%
from __future__ import annotations

import gc
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


PROJECT_ROOT = Path.cwd().resolve()
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "complete_evaluation"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass
class EvalConfig:
    base_model_path: Path = PROJECT_ROOT / "models" / "base" / "Qwen3.5-0.8B"
    finetuned_model_path: Path = PROJECT_ROOT / "models" / "qwen3.5-0.8b-chatdoctor-unsloth"
    test_data_path: Path = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
    train_log_path: Path = PROJECT_ROOT / "models" / "qwen3.5-0.8b-chatdoctor-unsloth" / "training_log.json"
    run_config_path: Path = PROJECT_ROOT / "models" / "qwen3.5-0.8b-chatdoctor-unsloth" / "run_config.json"
    output_dir: Path = OUTPUT_DIR
    max_new_tokens: int = env_int("EVAL_MAX_NEW_TOKENS", 160)
    max_input_length: int = env_int("EVAL_MAX_INPUT_LENGTH", 512)
    num_samples: Optional[int] = env_int("EVAL_NUM_SAMPLES", 25 if not env_flag("FULL_EVAL", False) else None)
    temperature: float = 0.0
    do_sample: bool = False
    seed: int = 42
    enable_bertscore: bool = False
    bertscore_model_type: str = "distilbert-base-uncased"


CONFIG = EvalConfig()

torch.manual_seed(CONFIG.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

print(f"Project root: {PROJECT_ROOT}")
print(f"Using device: {DEVICE}")
print(f"Using dtype: {DTYPE}")


# %% [markdown]
# ## 2. Paths and Environment Validation

# %%
def resolve_existing_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the candidate paths exist:\n" + "\n".join(str(path) for path in candidates)
    )


CONFIG.base_model_path = resolve_existing_path(
    CONFIG.base_model_path,
    PROJECT_ROOT / "models" / "base",
)
CONFIG.finetuned_model_path = resolve_existing_path(
    CONFIG.finetuned_model_path,
    PROJECT_ROOT / "model" / "qwen3.5-0.8b-chatdoctor-unsloth",
)
CONFIG.test_data_path = resolve_existing_path(
    CONFIG.test_data_path,
    PROJECT_ROOT / "data" / "processed" / "test.json",
)

CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

print("Base model path:", CONFIG.base_model_path)
print("Fine-tuned model path:", CONFIG.finetuned_model_path)
print("Test data path:", CONFIG.test_data_path)
print("Output dir:", CONFIG.output_dir)


# %% [markdown]
# ## 3. Prompt Templates

# %%
BASE_SYSTEM_PROMPT = (
    "You are a medical question answering assistant. Provide clear, cautious, "
    "general educational guidance and recommend professional care when needed."
)

PROMPT_BASELINE_SYSTEM_PROMPT = (
    "You are an experienced medical triage assistant. Answer in a cautious, "
    "structured way using this format: likely considerations, immediate home care, "
    "red flags, and when to see a doctor. Avoid definitive diagnosis, avoid "
    "inventing facts, and explicitly state uncertainty when the question lacks detail."
)


def combine_instruction_and_input(instruction: str, input_text: str = "") -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\nAdditional context:\n{input_text}"
    return instruction


def build_messages(question: str, system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]


# %% [markdown]
# ## 4. Dataset Loading

# %%
def load_json_dataset(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
        if isinstance(rows, dict):
            if "test" in rows:
                rows = rows["test"]
            else:
                raise ValueError("JSON file must contain a list or a dictionary with a 'test' key.")

    normalized_rows = []
    for row in rows:
        instruction = row.get("instruction", "")
        input_text = row.get("input", "")
        output_text = row.get("output", row.get("response", ""))
        normalized_rows.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "question": combine_instruction_and_input(instruction, input_text),
            }
        )
    return normalized_rows


test_rows = load_json_dataset(CONFIG.test_data_path)
if CONFIG.num_samples is not None:
    test_rows = test_rows[: CONFIG.num_samples]

print(f"Loaded {len(test_rows)} test samples.")
print("Example keys:", list(test_rows[0].keys()) if test_rows else [])


# %% [markdown]
# ## 5. Model Loading Utilities

# %%
def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def has_full_model_weights(model_path: Path) -> bool:
    return (model_path / "config.json").exists() and any(
        candidate.exists()
        for candidate in [
            model_path / "model.safetensors",
            model_path / "model.safetensors.index.json",
            model_path / "pytorch_model.bin",
        ]
    )


def resolve_adapter_path(model_path: Path) -> Optional[Path]:
    adapter_candidates = [
        model_path / "adapter",
        model_path,
    ]
    for candidate in adapter_candidates:
        if (candidate / "adapter_config.json").exists():
            return candidate
    return None


def load_tokenizer(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(model_path: Path):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


def load_base_model_and_tokenizer(config: EvalConfig):
    tokenizer = load_tokenizer(config.base_model_path)
    model = load_causal_lm(config.base_model_path)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def load_finetuned_model_and_tokenizer(config: EvalConfig):
    if has_full_model_weights(config.finetuned_model_path):
        tokenizer = load_tokenizer(config.finetuned_model_path)
        model = load_causal_lm(config.finetuned_model_path)
        model.to(DEVICE)
        model.eval()
        return model, tokenizer, "merged_full_model"

    adapter_path = resolve_adapter_path(config.finetuned_model_path)
    if adapter_path is None:
        raise FileNotFoundError(
            f"Could not find a merged model or adapter files under {config.finetuned_model_path}"
        )
    if PeftModel is None:
        raise ImportError("peft is required to load adapter-only fine-tuned weights.")

    tokenizer = load_tokenizer(adapter_path)
    base_model = load_causal_lm(config.base_model_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, "adapter_on_base"


# %% [markdown]
# ## 6. Generation Utilities

# %%
def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt_lines = []
    for message in messages:
        role = message["role"].upper()
        content = message["content"].strip()
        prompt_lines.append(f"{role}: {content}")
    prompt_lines.append("ASSISTANT:")
    return "\n".join(prompt_lines)


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> str:
    prompt = apply_chat_template(tokenizer, build_messages(question, system_prompt))
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

    outputs = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **({"temperature": temperature} if do_sample else {}),
    )

    prompt_length = encoded["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_length:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded.strip()


def run_inference_pass(
    rows: List[Dict[str, str]],
    model,
    tokenizer,
    label: str,
    system_prompt: str,
    config: EvalConfig,
) -> List[str]:
    predictions = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        prediction = generate_response(
            model=model,
            tokenizer=tokenizer,
            question=row["question"],
            system_prompt=system_prompt,
            max_input_length=config.max_input_length,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
        )
        predictions.append(prediction)
        if index == 1 or index % 25 == 0 or index == total:
            print(f"[{label}] Processed {index}/{total} samples")
    return predictions


# %% [markdown]
# ## 7. Metric Utilities

# %%
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def repetition_ratio(text: str) -> float:
    tokens = tokenize(text)
    if len(tokens) < 4:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    repeated = len(bigrams) - len(set(bigrams))
    return safe_divide(repeated, len(bigrams))


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n or n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n_score(prediction: str, reference: str, n: int) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    pred_ngrams = Counter(get_ngrams(pred_tokens, n))
    ref_ngrams = Counter(get_ngrams(ref_tokens, n))
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    overlap = pred_ngrams & ref_ngrams
    overlap_count = sum(overlap.values())
    precision = safe_divide(overlap_count, sum(pred_ngrams.values()))
    recall = safe_divide(overlap_count, sum(ref_ngrams.values()))
    return safe_divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0


def lcs_length(pred_tokens: List[str], ref_tokens: List[str]) -> int:
    if not pred_tokens or not ref_tokens:
        return 0
    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for j, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_score(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = safe_divide(lcs, len(pred_tokens))
    recall = safe_divide(lcs, len(ref_tokens))
    return safe_divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0


def sentence_bleu_score(
    prediction: str,
    reference: str,
    max_n: int = 4,
    smooth: float = 1.0,
) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(get_ngrams(pred_tokens, n))
        ref_ngrams = Counter(get_ngrams(ref_tokens, n))
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        precisions.append((overlap + smooth) / (total + smooth))

    if min(precisions) <= 0:
        return 0.0

    log_precision = sum(math.log(p) for p in precisions) / max_n
    brevity_penalty = 1.0
    if len(pred_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))
    return brevity_penalty * math.exp(log_precision)


def compute_metrics_bundle(
    predictions: List[str],
    references: List[str],
    enable_bertscore: bool = False,
    bertscore_model_type: str = "distilbert-base-uncased",
) -> Dict[str, float]:
    metrics = {
        "rouge1": float(
            sum(rouge_n_score(prediction, reference, 1) for prediction, reference in zip(predictions, references))
            / max(len(predictions), 1)
        ),
        "rouge2": float(
            sum(rouge_n_score(prediction, reference, 2) for prediction, reference in zip(predictions, references))
            / max(len(predictions), 1)
        ),
        "rougeL": float(
            sum(rouge_l_score(prediction, reference) for prediction, reference in zip(predictions, references))
            / max(len(predictions), 1)
        ),
        "bleu": float(
            sum(sentence_bleu_score(prediction, reference) for prediction, reference in zip(predictions, references))
            / max(len(predictions), 1)
        ),
        "token_f1": float(sum(token_f1(p, r) for p, r in zip(predictions, references)) / max(len(predictions), 1)),
        "avg_prediction_tokens": float(
            sum(len(tokenize(prediction)) for prediction in predictions) / max(len(predictions), 1)
        ),
        "avg_reference_tokens": float(
            sum(len(tokenize(reference)) for reference in references) / max(len(references), 1)
        ),
    }

    if enable_bertscore:
        try:
            from bert_score import score as bert_score

            _, _, f1 = bert_score(
                predictions,
                references,
                model_type=bertscore_model_type,
                lang="en",
                device=str(DEVICE),
                verbose=False,
            )
            metrics["bertscore_f1"] = float(f1.mean().item())
        except Exception as exc:
            print(f"BERTScore skipped: {exc}")

    return metrics


def score_prediction_rows(
    rows: List[Dict[str, str]],
    prediction_key: str,
) -> None:
    for row in rows:
        row[f"{prediction_key}_token_f1"] = token_f1(row[prediction_key], row["ground_truth"])
        row[f"{prediction_key}_repetition_ratio"] = repetition_ratio(row[prediction_key])


# %% [markdown]
# ## 8. Heuristic Error Analysis
#
# These labels are rule-based and should be treated as approximate diagnostics,
# not clinical truth.

# %%
MEDICAL_RISK_TERMS = {
    "insulin",
    "steroid",
    "antibiotic",
    "cancer",
    "tumor",
    "chemotherapy",
    "pregnancy",
    "stroke",
    "seizure",
    "dose",
    "dosage",
    "rabies",
    "heart attack",
}


def contains_phrase(text: str, phrases: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(phrase in normalized for phrase in phrases)


def classify_error(prediction: str, reference: str) -> str:
    prediction_norm = normalize_text(prediction)
    reference_norm = normalize_text(reference)
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    f1 = token_f1(prediction, reference)
    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    rep_ratio = repetition_ratio(prediction)

    overlap = set(pred_tokens) & set(ref_tokens)
    novel_ratio = safe_divide(len(set(pred_tokens) - overlap), len(set(pred_tokens)))

    if rep_ratio >= 0.2:
        return "repetition"

    if pred_len == 0 or pred_len <= max(8, math.floor(0.35 * ref_len)) or f1 < 0.18 and pred_len < ref_len * 0.5:
        return "incomplete_answer"

    risk_term_in_prediction = contains_phrase(prediction_norm, MEDICAL_RISK_TERMS)
    risk_term_missing_in_reference = risk_term_in_prediction and not contains_phrase(reference_norm, MEDICAL_RISK_TERMS)
    contradiction_cue = (
        ("not" in prediction_norm and "not" not in reference_norm)
        or ("immediate" in prediction_norm and "immediate" not in reference_norm and "emergency" not in reference_norm)
    )

    if (risk_term_missing_in_reference and f1 < 0.35) or contradiction_cue:
        return "wrong_medical_info"

    if novel_ratio > 0.72 and f1 < 0.25:
        return "hallucination"

    return "no_major_error"


def summarize_error_types(rows: List[Dict[str, str]], prediction_key: str) -> Dict[str, float]:
    counter = Counter(classify_error(row[prediction_key], row["ground_truth"]) for row in rows)
    total = max(len(rows), 1)
    summary = {label: round(100.0 * count / total, 2) for label, count in sorted(counter.items())}
    for label in ["hallucination", "incomplete_answer", "wrong_medical_info", "repetition", "no_major_error"]:
        summary.setdefault(label, 0.0)
    return summary


# %% [markdown]
# ## 9. Training Log and Overfitting Utilities

# %%
def load_training_log(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def extract_loss_points(log_history: List[Dict]) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    train_points = []
    eval_points = []
    for item in log_history:
        step = item.get("step")
        if step is None:
            continue
        if "loss" in item and "eval_loss" not in item:
            train_points.append((step, float(item["loss"])))
        if "eval_loss" in item:
            eval_points.append((step, float(item["eval_loss"])))
    return train_points, eval_points


def analyze_overfitting(log_history: List[Dict]) -> Dict[str, object]:
    train_points, eval_points = extract_loss_points(log_history)
    analysis = {
        "status": "not_available",
        "evidence": "Training log not found or missing validation loss.",
    }

    if not train_points or not eval_points:
        return analysis

    train_start = train_points[0][1]
    train_end = train_points[-1][1]
    eval_start = eval_points[0][1]
    eval_end = eval_points[-1][1]
    best_eval = min(loss for _, loss in eval_points)
    best_eval_step = min(eval_points, key=lambda item: item[1])[0]

    overfitting_detected = eval_end > best_eval + 0.03 and train_end < train_start

    if overfitting_detected:
        status = "possible_overfitting"
        evidence = (
            f"Training loss decreased from {train_start:.4f} to {train_end:.4f}, "
            f"but validation loss rose from its best value {best_eval:.4f} at step {best_eval_step} "
            f"to {eval_end:.4f} at the end."
        )
    else:
        status = "no_clear_overfitting"
        evidence = (
            f"Training loss decreased from {train_start:.4f} to {train_end:.4f}. "
            f"Validation loss also improved from {eval_start:.4f} to {eval_end:.4f}, "
            f"with the best validation loss {best_eval:.4f} at step {best_eval_step}."
        )

    analysis.update(
        {
            "status": status,
            "evidence": evidence,
            "train_start_loss": round(train_start, 4),
            "train_end_loss": round(train_end, 4),
            "eval_start_loss": round(eval_start, 4),
            "eval_end_loss": round(eval_end, 4),
            "best_eval_loss": round(best_eval, 4),
            "best_eval_step": int(best_eval_step),
        }
    )
    return analysis


# %% [markdown]
# ## 10. Run Generation
#
# To keep RAM usage manageable on CPU, the notebook evaluates the base model
# first, releases it, then loads the fine-tuned model.

# %%
ground_truths = [row["output"] for row in test_rows]

base_model, base_tokenizer = load_base_model_and_tokenizer(CONFIG)
base_predictions = run_inference_pass(
    rows=test_rows,
    model=base_model,
    tokenizer=base_tokenizer,
    label="base_model",
    system_prompt=BASE_SYSTEM_PROMPT,
    config=CONFIG,
)

prompt_baseline_predictions = run_inference_pass(
    rows=test_rows,
    model=base_model,
    tokenizer=base_tokenizer,
    label="prompt_baseline",
    system_prompt=PROMPT_BASELINE_SYSTEM_PROMPT,
    config=CONFIG,
)

del base_model
del base_tokenizer
clear_memory()

finetuned_model, finetuned_tokenizer, finetuned_loading_mode = load_finetuned_model_and_tokenizer(CONFIG)
finetuned_predictions = run_inference_pass(
    rows=test_rows,
    model=finetuned_model,
    tokenizer=finetuned_tokenizer,
    label="finetuned_model",
    system_prompt=BASE_SYSTEM_PROMPT,
    config=CONFIG,
)

del finetuned_model
del finetuned_tokenizer
clear_memory()

print("Finished all generation passes.")
print("Fine-tuned loading mode:", finetuned_loading_mode)


# %% [markdown]
# ## 11. Assemble Prediction Records

# %%
prediction_rows = []
for row, ground_truth, base_pred, prompt_pred, finetuned_pred in zip(
    test_rows,
    ground_truths,
    base_predictions,
    prompt_baseline_predictions,
    finetuned_predictions,
):
    prediction_rows.append(
        {
            "instruction": row["instruction"],
            "input": row["input"],
            "question": row["question"],
            "ground_truth": ground_truth,
            "base_prediction": base_pred,
            "prompt_baseline_prediction": prompt_pred,
            "finetuned_prediction": finetuned_pred,
        }
    )

score_prediction_rows(prediction_rows, "base_prediction")
score_prediction_rows(prediction_rows, "prompt_baseline_prediction")
score_prediction_rows(prediction_rows, "finetuned_prediction")

for row in prediction_rows:
    row["finetuned_minus_base_token_f1"] = (
        row["finetuned_prediction_token_f1"] - row["base_prediction_token_f1"]
    )
    row["finetuned_minus_prompt_baseline_token_f1"] = (
        row["finetuned_prediction_token_f1"] - row["prompt_baseline_prediction_token_f1"]
    )

print(f"Prepared {len(prediction_rows)} prediction records.")


# %% [markdown]
# ## 12. Compute Aggregate Metrics

# %%
metrics = {
    "dataset": {
        "test_path": str(CONFIG.test_data_path),
        "num_samples": len(prediction_rows),
    },
    "runtime": {
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "fine_tuned_loading_mode": finetuned_loading_mode,
    },
    "config": asdict(CONFIG),
    "models": {
        "base_model": compute_metrics_bundle(
            base_predictions,
            ground_truths,
            enable_bertscore=CONFIG.enable_bertscore,
            bertscore_model_type=CONFIG.bertscore_model_type,
        ),
        "prompt_baseline": compute_metrics_bundle(
            prompt_baseline_predictions,
            ground_truths,
            enable_bertscore=CONFIG.enable_bertscore,
            bertscore_model_type=CONFIG.bertscore_model_type,
        ),
        "finetuned_model": compute_metrics_bundle(
            finetuned_predictions,
            ground_truths,
            enable_bertscore=CONFIG.enable_bertscore,
            bertscore_model_type=CONFIG.bertscore_model_type,
        ),
    },
}

metrics["error_analysis"] = {
    "base_model": summarize_error_types(prediction_rows, "base_prediction"),
    "prompt_baseline": summarize_error_types(prediction_rows, "prompt_baseline_prediction"),
    "finetuned_model": summarize_error_types(prediction_rows, "finetuned_prediction"),
}

training_log = load_training_log(CONFIG.train_log_path)
metrics["overfitting_check"] = analyze_overfitting(training_log)

metrics["comparison_summary"] = {
    "finetuned_minus_base": {
        key: round(
            metrics["models"]["finetuned_model"][key] - metrics["models"]["base_model"][key],
            4,
        )
        for key in metrics["models"]["base_model"]
        if key in metrics["models"]["finetuned_model"]
    },
    "finetuned_minus_prompt_baseline": {
        key: round(
            metrics["models"]["finetuned_model"][key] - metrics["models"]["prompt_baseline"][key],
            4,
        )
        for key in metrics["models"]["prompt_baseline"]
        if key in metrics["models"]["finetuned_model"]
    },
}

print(json.dumps(metrics["models"], indent=2))


# %% [markdown]
# ## 13. Visualizations

# %%
def plot_metric_comparison(metrics_payload: Dict[str, Dict[str, float]], output_path: Path) -> None:
    tracked_metrics = ["rouge1", "rouge2", "rougeL", "bleu", "token_f1"]
    model_names = ["base_model", "prompt_baseline", "finetuned_model"]
    display_names = ["Base", "Prompt Baseline", "Fine-tuned"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = list(range(len(tracked_metrics)))
    width = 0.24

    for idx, (model_name, display_name) in enumerate(zip(model_names, display_names)):
        values = [metrics_payload[model_name].get(metric_name, 0.0) for metric_name in tracked_metrics]
        offsets = [x + (idx - 1) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=display_name)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tracked_metrics)
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison Across Evaluation Variants")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_loss_curves(log_history: List[Dict], output_path: Path) -> None:
    train_points, eval_points = extract_loss_points(log_history)
    if not train_points and not eval_points:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if train_points:
        xs, ys = zip(*train_points)
        ax.plot(xs, ys, label="Train loss", linewidth=2.0)
    if eval_points:
        xs, ys = zip(*eval_points)
        ax.plot(xs, ys, label="Validation loss", linewidth=2.0)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


metric_plot_path = CONFIG.output_dir / "metric_comparison.png"
loss_plot_path = CONFIG.output_dir / "loss_curves.png"

plot_metric_comparison(metrics["models"], metric_plot_path)
plot_loss_curves(training_log, loss_plot_path)

print("Saved metric plot to:", metric_plot_path)
print("Saved loss plot to:", loss_plot_path)


# %% [markdown]
# ## 14. Qualitative Analysis

# %%
def select_qualitative_examples(rows: List[Dict[str, str]], n_total: int = 10) -> List[Dict[str, str]]:
    sorted_rows = sorted(rows, key=lambda row: row["finetuned_minus_base_token_f1"], reverse=True)
    improved = sorted_rows[: n_total // 2]
    failed = list(reversed(sorted_rows[-(n_total - len(improved)) :]))
    selected = improved + failed

    deduplicated = []
    seen_questions = set()
    for row in selected:
        if row["question"] not in seen_questions:
            deduplicated.append(row)
            seen_questions.add(row["question"])

    if len(deduplicated) < n_total:
        for row in sorted_rows:
            if row["question"] not in seen_questions:
                deduplicated.append(row)
                seen_questions.add(row["question"])
            if len(deduplicated) == n_total:
                break
    return deduplicated[:n_total]


def describe_example(row: Dict[str, str]) -> str:
    improvement = row["finetuned_minus_base_token_f1"]
    if improvement > 0.05:
        note = "Fine-tuned model improved on token overlap and answer coverage."
    elif improvement < -0.05:
        note = "Fine-tuned model regressed relative to the base model."
    else:
        note = "Difference is small; inspect tone, caution, and factuality manually."
    return note


qualitative_examples = select_qualitative_examples(prediction_rows, n_total=10)

for idx, row in enumerate(qualitative_examples, start=1):
    print("=" * 120)
    print(f"Example {idx}")
    print("Instruction:", row["instruction"])
    if row["input"]:
        print("Input:", row["input"])
    print("\nGround truth:\n", row["ground_truth"])
    print("\nBase output:\n", row["base_prediction"])
    print("\nPrompt baseline output:\n", row["prompt_baseline_prediction"])
    print("\nFine-tuned output:\n", row["finetuned_prediction"])
    print("\nComment:", describe_example(row))
    print(
        "Scores:",
        {
            "base_token_f1": round(row["base_prediction_token_f1"], 4),
            "prompt_baseline_token_f1": round(row["prompt_baseline_prediction_token_f1"], 4),
            "finetuned_token_f1": round(row["finetuned_prediction_token_f1"], 4),
        },
    )


# %% [markdown]
# ## 15. Save Predictions and Reports

# %%
def build_report_text(metrics_payload: Dict[str, object], examples: List[Dict[str, str]]) -> str:
    models_metrics = metrics_payload["models"]
    overfit = metrics_payload["overfitting_check"]
    error_analysis = metrics_payload["error_analysis"]

    lines = []
    lines.append("Complete Evaluation Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Test samples evaluated: {metrics_payload['dataset']['num_samples']}")
    lines.append(f"Device: {metrics_payload['runtime']['device']}")
    lines.append(f"Fine-tuned loading mode: {metrics_payload['runtime']['fine_tuned_loading_mode']}")
    lines.append("")

    lines.append("Aggregate Metrics")
    lines.append("-" * 80)
    for model_name, scores in models_metrics.items():
        lines.append(model_name)
        for metric_name, value in scores.items():
            lines.append(f"  {metric_name}: {value:.4f}")
        lines.append("")

    lines.append("Error Analysis Summary (%)")
    lines.append("-" * 80)
    for model_name, summary in error_analysis.items():
        lines.append(model_name)
        for label, value in summary.items():
            lines.append(f"  {label}: {value:.2f}%")
        lines.append("")

    lines.append("Overfitting Check")
    lines.append("-" * 80)
    lines.append(f"Status: {overfit['status']}")
    lines.append(f"Evidence: {overfit['evidence']}")
    lines.append("")

    lines.append("Qualitative Review")
    lines.append("-" * 80)
    for idx, row in enumerate(examples, start=1):
        lines.append(f"Example {idx}")
        lines.append(f"Instruction: {row['instruction']}")
        if row["input"]:
            lines.append(f"Input: {row['input']}")
        lines.append(f"Ground truth: {row['ground_truth']}")
        lines.append(f"Base output: {row['base_prediction']}")
        lines.append(f"Prompt baseline output: {row['prompt_baseline_prediction']}")
        lines.append(f"Fine-tuned output: {row['finetuned_prediction']}")
        lines.append(f"Comment: {describe_example(row)}")
        lines.append("")

    return "\n".join(lines)


predictions_path = CONFIG.output_dir / "predictions.json"
metrics_path = CONFIG.output_dir / "metrics.json"
report_path = CONFIG.output_dir / "evaluation_report.txt"

predictions_path.write_text(json.dumps(prediction_rows, indent=2, ensure_ascii=False), encoding="utf-8")
metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
report_path.write_text(build_report_text(metrics, qualitative_examples), encoding="utf-8")

print("Saved predictions to:", predictions_path)
print("Saved metrics to:", metrics_path)
print("Saved report to:", report_path)


# %% [markdown]
# ## 16. Final Assignment Notes
#
# Recommended discussion points for the write-up:
# - Whether fine-tuning improves lexical overlap and answer completeness over the base model.
# - Whether the prompt-only baseline closes part of the gap without any parameter updates.
# - Whether the fine-tuned model is more cautious and domain-aligned on medical questions.
# - Whether the loss curve suggests healthy generalization or late-stage overfitting.
# - Which error types remain common after fine-tuning.
