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

# Load model from Hugging Face
model_name = "gizadatateam/ModernAraBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Test with Arabic text
text = "القاهرة هي [MASK] مصر"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Get prediction
masked_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = outputs.logits[0, masked_index].argmax(dim=-1)
predicted_word = tokenizer.decode(predicted_token_id)
print(f"Predicted: {predicted_word}")  # Expected: "عاصمة"
```

### 2. Fine-tune for Sentiment Analysis

```bash
python src/benchmarking/sa/sa_benchmark.py \
    --model-name modernarabert \
    --dataset hard \
    --batch-size 16 \
    --epochs 50 \
    --freeze
```

### 3. Run NER Benchmark

```bash
python src/benchmarking/ner/ner_benchmark.py \
    --model-name modernarabert \
    --batch-size 16 \
    --epochs 5 \
    --freeze
```

### 4. Pretrain from Scratch

```bash
# 1. Download and preprocess data
python scripts/pretraining/run_data_collection.py --config configs/pretraining_config.yaml

# 2. Extend tokenizer
python scripts/pretraining/run_tokenizer_extension.py --input-dir ./data/processed

# 3. Train
python scripts/pretraining/run_pretraining.py --config configs/pretraining_config.yaml
```

## What's Next?

- **Detailed Guides**: See [docs/](./docs/) for comprehensive documentation
  - [PRETRAINING.md](./docs/PRETRAINING.md): Complete pretraining guide
  - [BENCHMARKING.md](./docs/BENCHMARKING.md): Benchmark evaluation guide
  - [DATASETS.md](./docs/DATASETS.md): Dataset documentation
  - [MODEL_CARD.md](./docs/MODEL_CARD.md): Model specifications

- **Notebooks**: Explore [notebooks/](./notebooks/) for interactive examples

- **Configuration**: Customize [configs/](./configs/) for your experiments

- **Paper**: Read our [LREC 2026 paper](./original_code/ModernAraBert_LREC.pdf)

## Performance Highlights

| Task | Metric | ModernAraBERT | AraBERT v1 | mBERT |
|------|--------|---------------|------------|-------|
| **SA (HARD)** | Macro-F1 | **89.4%** | 72.7% | 71.7% |
| **NER** | Micro F1 | 82.1% | 78.9% | **90.7%** |
| **QA (ARCD)** | F1 / SM | **47.18 / 76.66** | 40.82 / 71.47 | 46.12 / 63.11 |

## Support

- **Issues**: [GitHub Issues](https://github.com/giza-data-team/ModernAraBERT/issues)
- **Email**: ahmed.aldamati@gizasystems.com
- **Model**: https://huggingface.co/gizadatateam/ModernAraBERT

## Citation

```bibtex
@inproceedings{eldamaty2026modernarabert,
  title={Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic},
  author={Eldamaty, Ahmed and Maher, Mohamed and Mostafa, Mohamed and Ashraf, Mariam and ElShawi, Radwa},
  booktitle={Proceedings of LREC-COLING 2026},
  year={2026}
}
```

---

Happy coding! 🚀

