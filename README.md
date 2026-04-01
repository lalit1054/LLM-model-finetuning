# Technical Assignment - Option 2: Medical Domain Q&A with ChatDoctor

This repository is structured as a submission-ready solution for **Challenge Option 2: Generative AI**. The selected task is **domain-specific Q&A in the medical domain** using the **ChatDoctor** dataset, with **QLoRA fine-tuning implemented using Unsloth**.

## Objective
Fine-tune a small language model for medical question answering so it produces more domain-adapted responses than the base model while staying within assignment constraints.

## Model Separation
The project keeps the original and fine-tuned artifacts separate:
- Base model snapshot: `models/base/Qwen3.5-0.8B`
- Fine-tuned model output: `models/qwen3.5-0.8b-chatdoctor-unsloth`

This prevents accidental overwriting of the downloaded base model and makes evaluation clearer.

## Chosen Setup
- Challenge option: `Option 2 - Generative AI`
- Task: `Domain-specific Q&A`
- Domain: `Medical`
- Dataset: `ChatDoctor`
- Local dataset file: `train-00000-of-00001-505f61796f2642f0.parquet`
- Base model path: `models/base/Qwen3.5-0.8B`
- Fine-tuned model path: `models/qwen3.5-0.8b-chatdoctor-unsloth`
- Model family: `Qwen3.5`
- Parameter count: `0.8B`
- Fine-tuning method: `QLoRA`
- Training framework: `Unsloth`

## Repository Layout
```text
.
├── README.md
├── requirements.txt
├── model_card.md
├── reports/
│   └── final_report_template.md
├── data/
│   ├── data_card.md
│   └── processed/
├── models/
│   ├── base/
│   │   └── Qwen3.5-0.8B/
│   └── qwen3.5-0.8b-chatdoctor-unsloth/
├── notebooks/
│   ├── 01_training_pipeline.ipynb
│   └── 02_evaluation.ipynb
├── outputs/
└── src/
    ├── data_prep.py
    ├── train.py
    └── evaluate.py
```

## 1. Prepare the ChatDoctor Dataset
```bash
python src/data_prep.py \
  --input_parquet train-00000-of-00001-505f61796f2642f0.parquet \
  --output_dir data/processed \
  --seed 42
```

## 2. Fine-Tune with Unsloth QLoRA
```bash
python src/train.py \
  --model_name models/base/Qwen3.5-0.8B \
  --train_file data/processed/train.jsonl \
  --validation_file data/processed/validation.jsonl \
  --output_dir models/qwen3.5-0.8b-chatdoctor-unsloth \
  --save_merged
```

## 3. Evaluate Base vs Fine-Tuned Model
```bash
python src/evaluate.py \
  --base_model models/base/Qwen3.5-0.8B \
  --adapter_path models/qwen3.5-0.8b-chatdoctor-unsloth \
  --test_file data/processed/test.jsonl \
  --output_dir outputs/eval \
  --num_samples 100
```

## Submission Template
Use:
- `reports/final_report_template.md`
- `outputs/eval/metrics.json`
- `outputs/eval/examples.jsonl`
- `outputs/eval/evaluation_report.md`

## Notes
- Do not save fine-tuned checkpoints inside `models/base/`.
- Keep the base model unchanged so comparisons remain reproducible.
- Training saves the LoRA adapter in `models/qwen3.5-0.8b-chatdoctor-unsloth/adapter/`.
- Passing `--save_merged` also writes a merged 16-bit fine-tuned model into `models/qwen3.5-0.8b-chatdoctor-unsloth/`.
