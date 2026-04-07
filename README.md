# Medical Q&A model finetune

This repository is for **Generative AI**. **domain-specific Q&A in the medical domain** using the **ChatDoctor** dataset, with **QLoRA fine-tuning implemented using Unsloth**.

## Objective
Fine-tune a small language model for medical question answering so it produces more domain-adapted responses than the base model while staying within assignment constraints.

## Model Separation
The project keeps the original and fine-tuned artifacts separate:
- Base model snapshot: `models/base/Qwen3.5-0.8B`
- Fine-tuned model output: `models/qwen3.5-0.8b-chatdoctor-unsloth`

This prevents accidental overwriting of the downloaded base model and makes evaluation clearer.

## Chosen Setup
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
├── 03_complete_evaluation_analysis.py
├── outputs/complete_evaluation
│   └── evaluation_report.txt
├── data/
│   ├── data_card.md
│   └── processed/
├── models/
│   ├── base/
│   │   └── Qwen3.5-0.8B/
│   └── qwen3.5-0.8b-chatdoctor-unsloth/
└── src/
    ├── data_prep.py
    ├── train.py
    
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
python 03_complete_evaluation_analysis.py \
  --base_model models/base/Qwen3.5-0.8B \
  --adapter_path models/qwen3.5-0.8b-chatdoctor-unsloth \
  --test_file data/processed/test.jsonl \
  --output_dir outputs/eval \
  --num_samples 100
```

.

## Notes
- Do not save fine-tuned checkpoints inside `models/base/`.

