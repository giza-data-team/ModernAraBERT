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

# Load model and tokenizer
model_name = "gizadatateam/ModernAraBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: Masked language modeling
text = "القاهرة هي [MASK] مصر"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

For sequence classification, NER, or QA tasks, see our [detailed examples](./notebooks/01_quick_start.ipynb).

---

## 📊 Performance

ModernAraBERT achieves state-of-the-art or competitive performance across three Arabic NLP tasks:

### Sentiment Analysis (Macro-F1 %)

| Dataset | AraBERT v1 | mBERT | ModernAraBERT | Improvement |
|---------|------------|-------|---------------|-------------|
| **AJGT** | 58.0 | 61.5 | **70.5** | +12.5% |
| **HARD** | 72.7 | 71.7 | **89.4** | **+16.7%** |
| **LABR** | 45.5 | 45.5 | **56.5** | +11.0% |

### Named Entity Recognition (Micro F1 %)

| Dataset | AraBERT v1 | mBERT | ModernAraBERT |
|---------|------------|-------|---------------|
| **ANERCorp** | 78.9 | **90.7** | 82.1 |

### Question Answering (ARCD Test %)

| Metric | AraBERT v1 | mBERT | ModernAraBERT | Improvement |
|--------|------------|-------|---------------|-------------|
| **Exact Match** | 13.26 | 15.27 | **18.73** | +41.3% |
| **F1-Score** | 40.82 | 46.12 | **47.18** | +15.6% |
| **Sentence Match** | 71.47 | 63.11 | **76.66** | +7.3% |

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

### Option 3: Docker

```bash
docker build -t modernarabert .
docker run -it modernarabert
```

---

## 📁 Repository Structure

```
ModernAraBERT/
├── src/                          # Source code modules
│   ├── pretraining/              # Pretraining scripts
│   ├── benchmarking/             # Benchmarking (SA, NER, QA)
│   └── utils/                    # Shared utilities
├── scripts/                      # Executable entry point
│   ├── pretraining/              # Pretraining pipelines
│   └── benchmarking/             # Benchmark evaluation
│       ├── run_sa_benchmark.py   # Streamlined SA benchmark interface
│       └── run_ner_benchmark.sh  # NER benchmark script
├── configs/                      # YAML configuration files
├── data/                         # Data download and preprocessing
├── docs/                         # Extended documentation
├── notebooks/                    # Jupyter notebook examples
├── tests/                        # Unit tests
├── results/                      # Benchmark results
└── original_code/                # Unmodified original implementation
```

---

## 🎯 Usage

### Pretraining

1. **Download and preprocess data**:
   ```bash
   python scripts/pretraining/run_data_collection.py --config configs/pretraining_config.yaml
   ```

2. **Extend tokenizer vocabulary**:
   ```bash
   python scripts/pretraining/run_tokenizer_extension.py --input-dir ./data/processed --output-tokenizer ./tokenizer
   ```

3. **Train the model**:
   ```bash
   python scripts/pretraining/run_pretraining.py --config configs/pretraining_config.yaml
   ```

See [PRETRAINING.md](./docs/PRETRAINING.md) for detailed instructions.

### Benchmarking

#### Sentiment Analysis

**Supported datasets**: HARD, LABR, AJGT

**New streamlined interface** (recommended):
```bash
# Run full pipeline on HARD dataset
python scripts/benchmarking/run_sa_benchmark.py --datasets hard

# Run full pipeline on multiple datasets
python scripts/benchmarking/run_sa_benchmark.py --datasets hard labr ajgt

# Run only data preparation stage
python scripts/benchmarking/run_sa_benchmark.py --stage prepare-data --datasets hard labr

# Run only benchmark stage (assumes data already prepared)
python scripts/benchmarking/run_sa_benchmark.py --stage benchmark --datasets hard

# Force re-download even if files exist
python scripts/benchmarking/run_sa_benchmark.py --datasets hard labr --force-redownload

# Custom model
python scripts/benchmarking/run_sa_benchmark.py --datasets hard --model-name arabert --model-path aubmindlab/bert-base-arabert
```

**Legacy shell script** (still supported):
```bash
bash scripts/benchmarking/run_sa_benchmark.sh --model-name modernbert --dataset hard
```

#### Named Entity Recognition

Run NER benchmarks:
```bash
bash scripts/benchmarking/run_ner_benchmark.sh --model-name modernbert
```

See [BENCHMARKING.md](./docs/BENCHMARKING.md) for detailed instructions and reproducing paper results.

---

## 📚 Documentation

- [**Pretraining Guide**](./docs/PRETRAINING.md): Complete guide to pretraining ModernAraBERT
- [**Benchmarking Guide**](./docs/BENCHMARKING.md): How to evaluate and reproduce paper results
- [**Dataset Documentation**](./docs/DATASETS.md): Information about training and evaluation datasets
- [**Model Card**](./docs/MODEL_CARD.md): Detailed model specifications and ethical considerations

---

## 🗂️ Datasets

### Pretraining Corpora
- **OSIAN**: Open Source International Arabic News
- **Arabic Billion Words**: 1.5B words Arabic corpus
- **Arabic Wikipedia**: Latest dump
- **OSCAR Arabic**: OSCAR 2022 Arabic subset

**Total**: ~17GB of preprocessed Arabic text, 6M+ sentences

### Benchmarking Datasets
- **Sentiment Analysis**: HARD, AJGT, LABR
- **Named Entity Recognition**: ANERCorp
- **Question Answering**: Arabic-SQuAD, ARCD

See [DATASETS.md](./docs/DATASETS.md) for download instructions and preprocessing details.

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

---

## 📄 Citation

If you use ModernAraBERT in your research, please cite:

```bibtex
@inproceedings{eldamaty2026modernarabert,
  title={Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic},
  author={Eldamaty, Ahmed and Maher, Mohamed and Mostafa, Mohamed and Ashraf, Mariam and ElShawi, Radwa},
  booktitle={Proceedings of the 2026 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2026)},
  year={2026},
  organization={ELRA}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library
- **Answer.AI** for the ModernBERT base model
- **Farasa** for Arabic text segmentation
- The Arabic NLP research community
- Giza Systems and University of Tartu

---

## 📧 Contact

- **Ahmed Eldamaty**: ahmed.aldamati@gizasystems.com
- **Mohamed Maher**: mohamed.abdelrahman@ut.ee
- **Radwa ElShawi**: radwa.elshawi@ut.ee

For issues and questions, please use the [GitHub issue tracker](https://github.com/giza-data-team/ModernAraBERT/issues).

---

<div align="center">

**[⬆ back to top](#modernarabert)**

Made with ❤️ by the Giza Data Team and University of Tartu

</div>

