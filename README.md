# 🧠 LLM Fine-Tuning with Qwen 3.5 using QLoRA (ChatDoctor Dataset)

This project focuses on fine-tuning the Qwen 3.5 using the ChatDoctor-7k dataset with QLoRA (Quantized Low-Rank Adaptation) via the Unsloth framework. The model is evaluated using standard NLP metrics such as BLEU and ROUGE.

## 🚀 Project Overview
Fine-tuning a medical-domain conversational LLM
Dataset: ChatDoctor-7k
Technique: QLoRA (efficient fine-tuning on limited GPU memory)
Framework: Unsloth (optimized for fast LLM training)
Evaluation Metrics: BLEU, ROUGE
## 📂 Project Structure
├── data/
│   └── chatdoctor_7k.json
├── notebooks/
│   └── training.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/
│   └── finetuned_qwen/
├── results/
│   ├── bleu_scores.json
│   └── rouge_scores.json
├── requirements.txt
└── README.md
🧾 Dataset
Name: ChatDoctor-7k
Domain: Medical Q&A
Format: Instruction-following conversational data

Each sample typically contains:

{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
## ⚙️ Methodology
🔹 Model
Base Model: Qwen 3.5
Type: Instruction-tuned LLM
🔹 Fine-Tuning Technique
QLoRA (Quantized Low-Rank Adaptation)
Reduces GPU memory usage
Enables training large models on limited hardware
🔹 Framework
Unsloth
Faster training
Memory-efficient
Optimized for LLM fine-tuning
🛠️ Installation
git clone https://github.com/your-username/qwen-qlora-chatdoctor.git
cd qwen-qlora-chatdoctor

pip install -r requirements.txt
## ▶️ Training
python src/train.py \
  --model_name qwen-3.5 \
  --dataset_path data/chatdoctor_7k.json \
  --output_dir models/finetuned_qwen \
  --batch_size 4 \
  --epochs 3
📊 Evaluation

Evaluation is done using:

BLEU Score → measures n-gram overlap
ROUGE Score → measures recall-based similarity
python src/evaluate.py \
  --model_path models/finetuned_qwen \
  --test_data data/test.json
📈 Results
Metric	Score
BLEU	XX.XX
ROUGE-1	XX.XX


## 💡 Key Learnings
Efficient fine-tuning of large models using QLoRA
Memory optimization using Unsloth
Handling domain-specific datasets (medical QA)
Evaluating generative models with BLEU & ROUGE
## ⚠️ Limitations
BLEU/ROUGE may not fully capture medical correctness
Requires domain-specific evaluation for real-world use
Limited dataset size (~7k samples)
## 🔮 Future Work
Use larger medical datasets
Integrate human evaluation
Deploy via API (FastAPI / Flask)
Add RAG (Retrieval-Augmented Generation)


## 📜 License

This project is licensed under the MIT License.

## 🙌 Acknowledgements
Qwen Model Team
ChatDoctor Dataset Contributors
Unsloth Framework

