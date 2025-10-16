# ModernAraBERT Benchmarking Guide

This guide explains how to evaluate ModernAraBERT on Arabic NLP benchmarks and provides concise CLI usage. For consolidated results, see [RESULTS.md](./RESULTS.md). For exact reproduction commands and seeds, see [REPRODUCIBILITY.md](./REPRODUCIBILITY.md).

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml && conda activate modernarabert
```

### Using YAML configs (optional)

Each runner in `scripts/benchmarking/` accepts `--config <yaml>` mirroring CLI flags. To dump defaults:

---

## Sentiment Analysis

Supported datasets: HARD, AJGT, LABR.

```bash
python src/benchmarking/sa/sa_benchmark.py --config configs/benchmarking/sa_benchmark.yaml
```

Run all sequentially (example):

```bash
for d in hard ajgt labr; do
  python scripts/benchmarking/run_sa_benchmark.py --config configs/benchmarking/sa_benchmark.yaml --datasets $d
done
```

---

## Named Entity Recognition

Dataset: ANERCorp (official splits via HF datasets).

```bash
python scripts/benchmarking/run_ner_benchmark.py --config configs/benchmarking/ner_benchmark.yaml
```

---

## Question Answering

Task: ARCD test; trained on Arabic-SQuAD + 50% ARCD.

```bash
python scripts/benchmarking/run_qa_benchmark.py --config configs/benchmarking/qa_benchmark.yaml
```

---
