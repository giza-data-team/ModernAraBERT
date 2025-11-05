# Reproducibility Guide

This guide provides the exact steps to reproduce the key results reported in the paper using pinned Python packages, fixed random seeds, and the scripts in this repository.

---

## Environment

- Python: 3.8.x
- Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:  
- Install PyTorch according to your platform/CUDA using the [official instructions](https://pytorch.org/get-started/locally/).
- We recommend a single GPU with ≥16GB VRAM. Numbers below were produced on an A100 40GB; smaller GPUs may require lower batch sizes.

## Seeds

Use a fixed seed for all runs:
- Global seeds: 42
- Components seeded in code: data loaders, NumPy, PyTorch CUDA and CPU RNGs

If you add your own scripts, ensure they set the same seed or pass `--seed 42` if available.

## Data

**Benchmark Datasets** (for reproducing paper tables):
- Datasets are downloaded automatically via [datasets](https://github.com/huggingface/datasets) where possible.
- After download/preprocessing, you should see counts matching the [DATASETS.md](./DATASETS.md).

Quick dataset preparation (sentiment analysis only, see [BENCHMARKING.md](./BENCHMARKING.md) for more details):
```bash
# Example: prepare SA data only (if the script supports staged runs)
python scripts/benchmarking/run_sa_benchmark.py --stage prepare-data --datasets hard ajgt labr
```

**Pretraining Data** (optional - see note below):
- Wikipedia dump links in `data/links.json` may become unavailable as Wikimedia archives old dumps
- If links fail, see [DATASETS.md](./DATASETS.md) for instructions on updating to current dumps
- **Important**: Different Wikipedia versions will produce different model weights
- To reproduce exact paper results, use the published `gizadatateam/ModernAraBERT` model from HuggingFace

## Commands to Reproduce Paper Tables

All commands below assume a fresh environment and no prior artifacts. Results are saved under `results/`.

### 1) Sentiment Analysis (Macro-F1)

Run each dataset:
```bash
# HARD
python scripts/benchmarking/run_sa_benchmark.py --config configs/benchmarking/sa_benchmark.yaml --datasets hard


# AJGT
python scripts/benchmarking/run_sa_benchmark.py --config configs/benchmarking/sa_benchmark.yaml --datasets ajgt

# LABR
python scripts/benchmarking/run_sa_benchmark.py --config configs/benchmarking/sa_benchmark.yaml --datasets labr
```

Expected metrics and dataset definitions are in `docs/RESULTS.md`.

### 2) Named Entity Recognition (Macro F1)

```bash
python scripts/benchmarking/run_ner_benchmark.py --config configs/benchmarking/ner_benchmark.yaml
```

### 3) Question Answering (ARCD: EM)

```bash
python scripts/benchmarking/run_qa_benchmark.py --config configs/benchmarking/qa_benchmark.yaml
```

## Pretraining (Optional)

To reproduce continued pretraining (not required for benchmark tables):
```bash
# Data collection and preprocessing
python scripts/pretraining/run_data_collection.py --config configs/pretraining/data_collection.yaml

# Tokenizer extension
python scripts/pretraining/run_tokenizer_extension.py --config configs/pretraining/tokenizer_extension.yaml

# Training
python scripts/pretraining/run_pretraining.py --config configs/pretraining/pretraining.yaml
```
See [PRETRAINING.md](./PRETRAINING.md) for details.

## Outputs and Verification

- Results are written under `results/` with per-run CSV/JSON logs.
- Compare against paper values in [RESULTS.md](./RESULTS.md). Small deviations (±0.5–1.0 points) can occur due to hardware/runtime differences.

## Hardware Notes

- If you hit CUDA OOM, reduce `--batch-size` or use gradient accumulation if supported.
- Mixed precision (fp16/bf16) is supported by Accelerate/Transformers and may improve throughput.

## Troubleshooting

- Ensure the model name is exactly `gizadatateam/ModernAraBERT` for inference/fine-tuning.
- Confirm that the encoder is frozen when the flag `--freeze` is provided.
- Verify dataset splits and preprocessing match [DATASETS.md](./DATASETS.md).

---

For questions, open an issue or contact the maintainers listed in [README.md](../README.md).


