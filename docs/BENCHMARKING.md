# ModernAraBERT Benchmarking Guide

This guide explains how to evaluate ModernAraBERT on Arabic NLP benchmarks and reproduce the results from our LREC 2026 paper.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Sentiment Analysis](#sentiment-analysis)
- [Named Entity Recognition](#named-entity-recognition)
- [Question Answering](#question-answering)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Adding Custom Benchmarks](#adding-custom-benchmarks)

---

## Overview

We evaluate ModernAraBERT on three core Arabic NLP tasks:

| Task | Datasets | Metric | ModernAraBERT | AraBERT v1 | mBERT |
|------|----------|--------|---------------|------------|-------|
| **Sentiment Analysis** | HARD, AJGT, LABR | Macro-F1 | **89.4%** (HARD) | 72.7% | 71.7% |
| **Named Entity Recognition** | ANERCorp | Micro F1 | 82.1% | 78.9% | **90.7%** |
| **Question Answering** | ARCD | EM / F1 / SM | **18.73 / 47.18 / 76.66** | 13.26 / 40.82 / 71.47 | 15.27 / 46.12 / 63.11 |

### Key Experimental Setup

- **Fine-tuning Strategy**: Frozen encoder + task-specific head only
- **Optimization**: AdamW with linear warmup
- **Early Stopping**: Patience = 10 epochs (SA), 3 epochs (NER)
- **Hardware**: NVIDIA A100 40GB

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate modernarabert
```

### Run All Benchmarks

```bash
# Sentiment Analysis (all datasets)
bash scripts/benchmarking/run_sa_benchmark.sh

# Named Entity Recognition
bash scripts/benchmarking/run_ner_benchmark.sh
```

---

## Sentiment Analysis

Evaluate on three Arabic sentiment datasets:

### Datasets

1. **HARD** (Hotel Arabic Reviews Dataset)
   - Binary sentiment (positive/negative)
   - Excludes neutral (3-star) reviews
   - Mix of MSA and dialectal Arabic
   - Train/Test split provided

2. **AJGT** (Arabic Jordanian General Tweets)
   - 1,800 labeled tweets
   - Binary sentiment
   - 60/20/20 train/val/test split

3. **LABR** (Large-Scale Arabic Book Reviews)
   - Book reviews
   - Unbalanced binary version
   - Large-scale dataset

### Running SA Benchmarks

#### Individual Dataset

```bash
python src/benchmarking/sa/sa_benchmark.py \
    --model-name modernarabert \
    --dataset hard \
    --batch-size 16 \
    --epochs 50 \
    --learning-rate 2e-5 \
    --freeze
```

#### All Datasets (Sequential)

```bash
# HARD
python src/benchmarking/sa/sa_benchmark.py \
    --model-name modernarabert \
    --dataset hard \
    --benchmark

# AJGT
python src/benchmarking/sa/sa_benchmark.py \
    --model-name modernarabert \
    --dataset ajgt \
    --benchmark

# LABR
python src/benchmarking/sa/sa_benchmark.py \
    --model-name modernarabert \
    --dataset labr \
    --benchmark
```

### Configuration

Customize `configs/sa_benchmark_config.yaml`:

```yaml
model:
  model_name: "modernarabert"
  freeze_encoder: true

training:
  learning_rate: 2.0e-5
  batch_size: 16
  num_epochs: 50
  early_stopping:
    patience: 10

evaluation:
  metrics:
    - "macro_f1"  # Primary metric
```

### Expected Results

| Dataset | Metric | ModernAraBERT | AraBERT v1 | mBERT |
|---------|--------|---------------|------------|-------|
| AJGT | Macro-F1 | **70.5%** | 58.0% | 61.5% |
| HARD | Macro-F1 | **89.4%** | 72.7% | 71.7% |
| LABR | Macro-F1 | **56.5%** | 45.5% | 45.5% |

---

## Named Entity Recognition

Evaluate on ANERCorp dataset.

### Dataset

- **ANERCorp**: Arabic Named Entity Recognition corpus
- **Entity Types**: Person (PER), Location (LOC), Organization (ORG), Miscellaneous (MISC)
- **Tagging Scheme**: IOB2 format
- **Splits**: Official CAMeL Lab splits via Hugging Face

### Running NER Benchmark

```bash
python src/benchmarking/ner/ner_benchmark.py \
    --model-name modernarabert \
    --dataset anercorp \
    --batch-size 16 \
    --epochs 5 \
    --learning-rate 2e-5 \
    --freeze
```

Or using the shell script:

```bash
bash scripts/benchmarking/run_ner_benchmark.sh --model-name modernarabert
```

### Labeling Strategy

**First-Subtoken-Only Labeling**:
- First subtoken of each word gets the gold label
- Continuation subtokens are either:
  - Mapped to I-label (e.g., B-PER → I-PER)
  - Or masked with -100 (ignored in loss)

Example:
```
Word: محمد → [محـ, ـمد]
Labels: [B-PER, I-PER] or [B-PER, -100]
```

### Configuration

Customize `configs/ner_benchmark_config.yaml`:

```yaml
model:
  model_name: "modernarabert"
  freeze_encoder: true

dataset:
  tagging_scheme: "IOB2"
  subword_labeling: "first_token_only"

training:
  learning_rate: 2.0e-5
  batch_size: 16
  num_epochs: 5
  early_stopping:
    patience: 3

evaluation:
  metrics:
    - "micro_f1"  # Primary metric
    - "per_entity_f1"
```

### Expected Results

| Model | Micro F1 | Notes |
|-------|----------|-------|
| **mBERT** | **90.7%** | Best performance |
| **ModernAraBERT** | 82.1% | Better than AraBERT |
| AraBERT v1 | 78.9% | Baseline |

**Analysis**: mBERT's superior performance suggests multilingual pretraining benefits token-level tasks, though ModernAraBERT improves over AraBERT.

---

## Question Answering

Evaluate on Arabic Reading Comprehension Dataset (ARCD).

### Dataset

- **Training**: Arabic-SQuAD + 50% of ARCD
- **Testing**: Remaining 50% of ARCD
- **Task**: Extractive span-based QA
- **Metrics**:
  - **EM (Exact Match)**: Exact string match
  - **F1**: Token-level overlap
  - **SM (Sentence Match)**: Sentence-level semantic alignment

### Running QA Benchmark

```bash
python src/benchmarking/qa/qa_benchmark.py \
    --model-name modernarabert \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 3e-5 \
    --max-length 512 \
    --doc-stride 128 \
    --freeze
```

### Configuration

Key hyperparameters:

```yaml
model:
  model_name: "modernarabert"
  freeze_encoder: true

training:
  learning_rate: 3.0e-5
  batch_size: 32  # ModernAraBERT, 64 for AraBERT
  num_epochs: 200
  early_stopping:
    patience: 10

preprocessing:
  max_length: 512
  doc_stride: 128  # For long contexts
```

### Expected Results

| Metric | ModernAraBERT | AraBERT v1 | mBERT | Improvement |
|--------|---------------|------------|-------|-------------|
| **Exact Match** | **18.73%** | 13.26% | 15.27% | **+41.3%** |
| **F1-Score** | **47.18%** | 40.82% | 46.12% | **+15.6%** |
| **Sentence Match** | **76.66%** | 71.47% | 63.11% | **+7.3%** |

**Analysis**: ModernAraBERT excels at semantic alignment (high SM), demonstrating strong understanding of longer-span contexts.

---

## Reproducing Paper Results

### Complete Reproduction Script

```bash
#!/bin/bash
# reproduce_paper_results.sh

echo "===== Reproducing LREC 2026 Paper Results ====="

# 1. Sentiment Analysis
echo "\n[1/3] Running Sentiment Analysis Benchmarks..."

python src/benchmarking/sa/sa_benchmark.py \
    --model-name modernarabert \
    --dataset hard \
    --batch-size 16 \
    --epochs 50 \
    --learning-rate 2e-5 \
    --freeze \
    --benchmark

python src/benchmarking/sa/sa_benchmark.py \
    --model-name modernarabert \
    --dataset ajgt \
    --batch-size 16 \
    --epochs 50 \
    --learning-rate 2e-5 \
    --freeze \
    --benchmark

python src/benchmarking/sa/sa_benchmark.py \
    --model-name modernarabert \
    --dataset labr \
    --batch-size 16 \
    --epochs 50 \
    --learning-rate 2e-5 \
    --freeze \
    --benchmark

# 2. Named Entity Recognition
echo "\n[2/3] Running NER Benchmark..."

python src/benchmarking/ner/ner_benchmark.py \
    --model-name modernarabert \
    --batch-size 16 \
    --epochs 5 \
    --learning-rate 2e-5 \
    --freeze

# 3. Question Answering
echo "\n[3/3] Running QA Benchmark..."

python src/benchmarking/qa/qa_benchmark.py \
    --model-name modernarabert \
    --batch-size 32 \
    --epochs 200 \
    --learning-rate 3e-5 \
    --freeze

echo "\n===== Reproduction Complete ====="
echo "Results saved to ./results/"
```

### Comparing with Baselines

Run for all models:

```bash
# ModernAraBERT
bash reproduce_paper_results.sh --model modernarabert

# AraBERT v1
bash reproduce_paper_results.sh --model arabert

# mBERT
bash reproduce_paper_results.sh --model mbert
```

### Verification

Compare your results with paper results in `results/`:

```bash
python scripts/compare_results.py \
    --paper-results results/paper_results.csv \
    --your-results results/modernarabert_*.csv
```

---

## Resource Usage Analysis

### Memory Benchmarks

Expected resource usage (from paper):

#### NER Task

| Model | Peak RAM (GB) | Peak VRAM (GB) | Throughput (samples/sec) |
|-------|---------------|----------------|--------------------------|
| AraBERT | 1.53 | 0.52 | 925.4 |
| mBERT | 1.60 | 0.68 | - |
| ModernAraBERT | 1.49 | 0.83 | 495.8 |

#### QA Task

| Model | Peak RAM (GB) | Peak VRAM (GB) |
|-------|---------------|----------------|
| AraBERT | 1.42 | 2.07 |
| mBERT | 1.46 | 2.84 |
| ModernAraBERT | 1.39 | 3.22 |

#### SA Task

| Model | Peak RAM (GB) | Peak VRAM (GB) |
|-------|---------------|----------------|
| AraBERT | 1.65 | 0.52 |
| mBERT | 1.66 | 0.68 |
| ModernAraBERT | 1.36 | 0.82 |

**Trade-off**: ModernAraBERT uses more VRAM but achieves higher accuracy.

---

## Adding Custom Benchmarks

### 1. Add Dataset Configuration

Edit `src/benchmarking/sa/datasets.py` (for SA) or equivalent:

```python
def load_custom_dataset(data_path, split="train"):
    """Load custom Arabic sentiment dataset."""
    # Your implementation
    return dataset
```

### 2. Update Configuration

Add to `configs/sa_benchmark_config.yaml`:

```yaml
dataset:
  datasets:
    custom:
      source: "local"
      data_path: "./data/custom_dataset"
      num_classes: 3
```

### 3. Run Benchmark

```bash
python src/benchmarking/sa/sa_benchmark.py \
    --model-name modernarabert \
    --dataset custom \
    --config configs/sa_benchmark_config.yaml
```

---

## Troubleshooting

### Low Performance

**Problem**: Results significantly lower than paper

**Solutions**:
1. Verify correct model: `gizadatateam/ModernAraBERT`
2. Check frozen encoder: `--freeze` flag
3. Verify hyperparameters match config
4. Ensure correct dataset preprocessing

### Out of Memory

**Problem**: CUDA OOM during benchmarking

**Solutions**:
1. Reduce batch size:
   ```bash
   --batch-size 8  # Instead of 16
   ```

2. Use gradient accumulation:
   ```bash
   --batch-size 8 --gradient-accumulation-steps 2
   ```

3. Reduce sequence length (if applicable)

### Slow Training

**Problem**: Training takes too long

**Solutions**:
1. Enable mixed precision:
   ```yaml
   training:
     mixed_precision: "fp16"
   ```

2. Increase num_workers:
   ```yaml
   system:
     num_workers: 4
   ```

3. Use SSD for data storage

---

## Citation

If you use these benchmarks, please cite:

```bibtex
@inproceedings{eldamaty2026modernarabert,
  title={Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic},
  author={Eldamaty, Ahmed and Maher, Mohamed and Mostafa, Mohamed and Ashraf, Mariam and ElShawi, Radwa},
  booktitle={Proceedings of LREC-COLING 2026},
  year={2026}
}
```

---

## Support

For questions or issues:
- GitHub Issues: [ModernAraBERT/issues](https://github.com/giza-data-team/ModernAraBERT/issues)
- Email: ahmed.aldamati@gizasystems.com

