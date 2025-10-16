# Pretraining (Concise)

Minimal steps to reproduce ModernAraBERT pretraining. For exact seeds, pinned env, and timings see `docs/REPRODUCIBILITY.md`.

## What you need
- Python 3.8
- 1× GPU (≥16GB VRAM recommended)
- Base: `answerdotai/ModernBERT-base`

Install:
```bash
pip install -r requirements.txt
```

## 1) Data (download + preprocess)
```bash
python scripts/pretraining/run_data_collection.py --config configs/pretraining/data_collection.yaml
```
- Sources: OSIAN, Arabic Billion Words, Arabic Wikipedia, OSCAR (≈9.8GB raw data).
- See brief dataset notes in [DATASETS.md](./DATASETS.md).

## 2) Extend tokenizer (Arabic tokens)
```bash
python scripts/pretraining/run_tokenizer_extension.py --config configs/pretraining/tokenizer_extension.yaml
```

## 3) Train
```bash
python scripts/pretraining/run_pretraining.py --config configs/pretraining/pretraining.yaml
```

Links: [REPRODUCIBILITY.md](./REPRODUCIBILITY.md) (exact commands) · [RESULTS.md](./RESULTS.md) (paper tables) · [DATASETS.md](./DATASETS.md) (versions)

