# ModernAraBERT

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ModernAraBERT-blue)](https://huggingface.co/gizadatateam/ModernAraBERT)
[![Paper](https://img.shields.io/badge/Paper-LREC%202026-red)](https://github.com/giza-data-team/ModernAraBERT)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Efficient Adaptation of English Language Models for Arabic NLP**

[Paper](https://github.com/giza-data-team/ModernAraBERT) | [Model](https://huggingface.co/gizadatateam/ModernAraBERT) | [Documentation](./docs/) | [Datasets](./docs/DATASETS.md)

</div>

---

## 📖 Overview

**ModernAraBERT** is a resource-efficient adaptation of the English-pretrained [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) model to Arabic. Our approach leverages continued pretraining on large curated Arabic corpora, followed by lightweight task-specific fine-tuning with frozen encoder backbones. This strategy preserves cross-lingual knowledge while effectively capturing Arabic morphology, offering a practical alternative to training monolingual models from scratch.

### Key Features

- 🚀 **Superior Performance**: Consistent improvements over AraBERT v1, with gains up to **+17% in sentiment analysis**
- 💾 **Resource Efficient**: Continued pretraining instead of training from scratch
- 🌍 **Cross-lingual Transfer**: Leverages knowledge from English-pretrained ModernBERT
- 📊 **Comprehensive Evaluation**: Tested on sentiment analysis, NER, and question answering
- 🔧 **Easy to Use**: Pre-trained model available on Hugging Face

---

## 🚀 Quick Start

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

For task-specific examples, see [QUICK_START.md](./QUICK_START.md).

---

## 📊 Performance

See consolidated tables in [docs/RESULTS.md](./docs/RESULTS.md). Reproduction commands are in [docs/REPRODUCIBILITY.md](./docs/REPRODUCIBILITY.md).

---

## 🔧 Installation

### Option 1: pip (Recommended)

```bash
pip install -r requirements.txt
```

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate modernarabert
```

---

## 📁 Next Steps

- Read the [Quick Start](./QUICK_START.md) to run inference and a minimal fine-tune.
- See [Reproducibility](./docs/REPRODUCIBILITY.md) to reproduce paper tables.

---

## 📚 Documentation

- [**Quick Start**](./QUICK_START.md)
- [**Reproducibility**](./docs/REPRODUCIBILITY.md)
- [**Results**](./docs/RESULTS.md)
- [**Pretraining Guide**](./docs/PRETRAINING.md)
- [**Benchmarking Guide**](./docs/BENCHMARKING.md)
- [**Dataset Documentation**](./docs/DATASETS.md)
- [**Model Card**](./docs/MODEL_CARD.md)

---

## 🏗️ Model Architecture

ModernAraBERT is based on **ModernBERT-base** with the following specifications:

- **Layers**: 22 transformer layers
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Parameters**: ~149M (base) + extended vocabulary
- **Vocabulary**: Original ModernBERT + **80K Arabic tokens**
- **Max Sequence Length**: 512 tokens (training), 8192 (architecture support)
- **Position Embeddings**: Rotary Position Embeddings (RoPE)
- **Activation**: GeGLU
- **Attention**: Alternating global-local attention

<!--
## 📄 Citation

If you use ModernAraBERT in your research, please cite:

```bibtex
@inproceedings{<paper_id>,
}

---
-->

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

<div align="center">

**[⬆ back to top](#modernarabert)**

</div>

