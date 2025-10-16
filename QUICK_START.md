# ModernAraBERT - Quick Start Guide

Get started with ModernAraBERT in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/giza-data-team/ModernAraBERT.git
cd ModernAraBERT

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate modernarabert
```

## Quick Usage Examples

### 1. Load Pre-trained Model (Inference)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load model and tokenizer
model_name = "gizadatateam/ModernAraBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: Masked language modeling
text = "القاهرة هي [MASK] مصر"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Find index of [MASK] token
mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()

# Get logits for [MASK] position
mask_logits = outputs.logits[0, mask_token_index]

# Get predicted token id (highest probability)
predicted_token_id = mask_logits.argmax(dim=-1).item()

# Decode predicted word
predicted_word = tokenizer.decode([predicted_token_id]).strip()
print(f"Predicted: {predicted_word}")
```

### 2. Minimal Fine-tune for Sentiment Analysis

```bash
python src/benchmarking/sa/sa_benchmark.py --config configs/benchmarking/sa_benchmark.yaml
```

### 3. Minimal NER Benchmark

```bash
python src/benchmarking/ner/ner_benchmark.py --config configs/benchmarking/ner_benchmark.yaml
```

### 4. Minimal QA Benchmark

```bash
python src/benchmarking/qa/qa_benchmark.py --config configs/benchmarking/qa_benchmark.yaml
```

### 5. (Optional) Pretrain

```bash
# 1. Download the data
python scripts/pretraining/run_data_collection.py --config configs/pretraining/data_collection.yaml

# 2. Preprocess the data
python scripts/pretraining/run_data_preprocessing.py --config configs/pretraining/data_preprocessing.yaml

# 3. Extend tokenizer
python scripts/pretraining/run_tokenizer_extension.py --config configs/pretraining/tokenizer_extension.yaml

# 4. Train
python scripts/pretraining/run_pretraining.py --config configs/pretraining/pretraining.yaml
```

## What's Next?

- Detailed guides: see [docs/](./docs/)

  - [REPRODUCIBILITY.md](./docs/REPRODUCIBILITY.md): exact steps to reproduce results
  - [RESULTS.md](./docs/RESULTS.md): consolidated paper tables
  - [PRETRAINING.md](./docs/PRETRAINING.md): pretraining details
  - [BENCHMARKING.md](./docs/BENCHMARKING.md): task-specific usage
  - [DATASETS.md](./docs/DATASETS.md): dataset details

- Notebooks: explore [notebooks/](./notebooks/)
- Configs: customize [configs/](./configs/)

## Support

- Issues: https://github.com/giza-data-team/ModernAraBERT/issues
