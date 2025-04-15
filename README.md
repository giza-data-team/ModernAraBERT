```
```markdown
# Arabic ModernBERT

This repository provides scripts and resources for **pretraining** and **benchmarking** a ModernBERT model designed for the Arabic language. It covers tasks like sentiment analysis (SA), named entity recognition (NER), and question answering (QA), along with tools to train and extend the model.

---

## 📁 Repository Structure

```
.
├── Benchmarking/
│   ├── Run AraBERT SA_Benchmarks.sh
│   ├── Run ModernBERT SA_Benchmarks.sh
│   ├── ner_benchmarking.py
│   ├── qa_benchmarking.py
│   ├── run_qa_benchmarks.sh
│   └── sa_benchmarking.py
│
└── Pretraining/
    ├── Data collection and preprocessing ....py
    ├── ModernBERT Training.py
    ├── Tokenizer vocab extending.py
    └── links.json
```

---

## 🚀 Getting Started

### 1. Environment Setup

Ensure you have the required dependencies installed (adjust as needed for your environment):

```bash
pip install transformers datasets torch sentencepiece scikit-learn
```

---

## 🧪 Benchmarking

### 1. Sentiment Analysis

- **ModernBERT**:
```bash
bash Run\ ModernBERT\ SA_Benchmarks.sh
```

- **AraBERT (for comparison)**:
```bash
bash Run\ AraBERT\ SA_Benchmarks.sh
```

- Direct Python usage:
```bash
python sa_benchmarking.py --model-name modernbert --epochs 5 --batch-size 16
```

### 2. Named Entity Recognition (NER)

```bash
python ner_benchmarking.py --model-name modernbert --dataset ner_dataset_path ...
```

### 3. Question Answering (QA)

- Using the Python script:
```bash
python qa_benchmarking.py --model-name modernbert --epochs 3 --max-length 512 ...
```

- Or with the shell script:
```bash
bash run_qa_benchmarks.sh
```

---

## 🔧 Pretraining

### 1. Prepare the Dataset

- Update the `links.json` file with links to large Arabic corpora.
- Run the data preprocessing script:

```bash
python "Data collection and preprocessing ....py" --output-dir ./datasets/
```

### 2. Extend Tokenizer (Optional)

If you need to add domain-specific vocabulary:

```bash
python "Tokenizer vocab extending.py" --input-dir ./datasets --output-tokenizer ./Tokenizer
```

### 3. Train ModernBERT

```bash
python "ModernBERT Training.py" \
  --train-dir ./datasets/train \
  --val-dir ./datasets/val \
  --epochs 5 \
  --batch-size 32 \
  --learning-rate 1e-4
```

---

## 💡 Tips & Best Practices

- Use GPU for faster training and benchmarking.
- Enable mixed precision (`fp16`) for better performance.
- Customize hyperparameters (`batch-size`, `epochs`, etc.) for your needs.
- Monitor logs for loss/accuracy trends during training and evaluation.
- Ensure dataset diversity to generalize well in Arabic NLP tasks.

---

## 🤝 Contributing

We welcome contributions!

- 🐛 Report bugs via Issues.
- 🔧 Submit enhancements via Pull Requests.
- 💬 Join Discussions if available.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Hugging Face Transformers
- Farasa
- AraBERT
- The Arabic NLP Community

---

```

Let me know if you'd like me to export this as a file or format it for GitHub directly!
