# Arabic ModernBERT

Arabic ModernBERT is a powerful Arabic language model trained on large-scale Arabic corpora. This repository includes tools for both **pretraining** the model and **benchmarking** it on common NLP tasks like Sentiment Analysis (SA), Named Entity Recognition (NER), and Question Answering (QA).

---

## 📁 Repository Structure

```
Arabic-ModernBERT/
├── Benchmarking/
│   ├── Run AraBERT SA_Benchmarks.sh
│   ├── Run ModernBERT SA_Benchmarks.sh
│   ├── ner_benchmarking.py
│   ├── qa_benchmarking.py
│   ├── run_qa_benchmarks.sh
│   └── sa_benchmarking.py
│
├── Pretraining/
│   ├── Data collection and preprocessing ....py
│   ├── ModernBERT Training.py
│   ├── Tokenizer vocab extending.py
│   └── links.json
│
└── README.md
```

---

## ✅ Requirements

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Farasa (Arabic segmenter)
- tqdm, scikit-learn, pandas, numpy

Install requirements with:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

### 🔧 Pretraining ModernBERT

1. **Data Collection & Preprocessing**
```bash
python "Pretraining/Data collection and preprocessing ....py" --output-dir ./datasets/
```
Make sure `links.json` includes the datasets' URLs.

2. **Extend Tokenizer (Optional)**
```bash
python "Pretraining/Tokenizer vocab extending.py" --input-dir ./datasets --output-tokenizer ./Tokenizer
```

3. **Train the Model**
```bash
python "Pretraining/ModernBERT Training.py" \
  --train-dir ./datasets/train \
  --val-dir ./datasets/val \
  --epochs 5 \
  --batch-size 32 \
  --learning-rate 1e-4
```

---

### 🧪 Benchmarking the Model

#### 1. Sentiment Analysis (SA)

- Run ModernBERT:
```bash
bash Benchmarking/Run\ ModernBERT\ SA_Benchmarks.sh
```

- Run AraBERT for comparison:
```bash
bash Benchmarking/Run\ AraBERT\ SA_Benchmarks.sh
```

- Or use Python directly:
```bash
python Benchmarking/sa_benchmarking.py --model-name modernbert --epochs 10 --batch-size 16
```

#### 2. Named Entity Recognition (NER)
```bash
python Benchmarking/ner_benchmarking.py --model-name modernbert --dataset path_to_ner_dataset --batch-size 16
```

#### 3. Question Answering (QA)

- With Python:
```bash
python Benchmarking/qa_benchmarking.py --model-name modernbert --epochs 3 --max-length 512 --doc-stride 128
```

- Or using shell script:
```bash
bash Benchmarking/run_qa_benchmarks.sh
```

---

### 💬 Inference Example

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="./model_checkpoints/checkpoint_step_13000/", tokenizer="./Tokenizer")
question = "ما هي عاصمة مصر؟"
context = "القاهرة هي عاصمة مصر وتعتبر من أكبر المدن في إفريقيا."
result = qa_pipeline({"question": question, "context": context})
print(result)
```

---

## 📌 Notes & Tips

- Use GPU for faster training & inference
- Use `fp16` training to save memory
- Monitor checkpoints/logs regularly
- Use high-quality, diverse Arabic corpora

---

## 📄 License
This project is licensed under the MIT License.

---

## 🙌 Acknowledgments
- Hugging Face
- Farasa
- AraBERT
- Arabic NLP research community

---

> Created and maintained by Giza Data Team
