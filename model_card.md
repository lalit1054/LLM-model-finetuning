# Model Card: Qwen3.5-0.8B ChatDoctor Unsloth QLoRA

## Artifact Separation
- Base model snapshot: `models/base/Qwen3.5-0.8B`
- Fine-tuned adapter: `models/adapters/qwen3.5-0.8b-chatdoctor-unsloth`

The fine-tuned model is stored separately from the original base model so the downloaded checkpoint remains unchanged.

## Model Details
- Base model path: `models/base/Qwen3.5-0.8B`
- Fine-tuned adapter path: `models/adapters/qwen3.5-0.8b-chatdoctor-unsloth`
- Model family: `Qwen3.5`
- Parameter count: `0.8B`
- Adaptation method: `QLoRA`
- Training framework: `Unsloth`
- Domain: Medical Q&A

## Training Data
- Dataset: ChatDoctor
- Local source: `train-00000-of-00001-505f61796f2642f0.parquet`
- Processed with: `python src/data_prep.py`
- Format used for training: single-turn instruction-response medical Q&A

## LoRA Configuration
- Rank (`r`): `16`
- Alpha: `32`
- Dropout: `0.0`
- Quantization: 4-bit loading for QLoRA
- Target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

## Training Setup
- Framework: Unsloth + TRL `SFTTrainer`
- Epochs: `2`
- Learning rate: `2e-4`
- Batch size per device: `1`
- Gradient accumulation steps: `16`
- Max sequence length: `768`
- Precision: `bf16`

## Fine-Tuning Command
```bash
python src/train.py   --model_name models/base/Qwen3.5-0.8B   --train_file data/processed/train.jsonl   --validation_file data/processed/validation.jsonl   --output_dir models/adapters/qwen3.5-0.8b-chatdoctor-unsloth
```

## Evaluation Command
```bash
python src/evaluate.py   --base_model models/base/Qwen3.5-0.8B   --adapter_path models/adapters/qwen3.5-0.8b-chatdoctor-unsloth   --test_file data/processed/test.jsonl   --output_dir outputs/eval   --num_samples 100
```
