# Datasets

Sources and settings used in ModernAraBERT. For detailed reproduction and counts, see [REPRODUCIBILITY.md](./REPRODUCIBILITY.md).

Pipeline (download + preprocess):

```bash
# Download and extract all datasets to data/raw
python scripts/pretraining/run_data_collection.py --config configs/pretraining/data_collection.yaml

# Preprocess the data and save to data/processed
python scripts/pretraining/run_data_preprocessing.py --config configs/pretraining/data_preprocessing.yaml
```

## Benchmarks

- Sentiment (Macro-F1): HARD (binary, no 3-star), AJGT (1,800 tweets), LABR (binary, std split)
- NER (Macro-F1): ANERCorp (official CAMeL splits)
- QA (EM on ARCD test): Train on Arabic-SQuAD + 50% ARCD; test on remaining 50% ARCD

Links: [RESULTS.md](./RESULTS.md) (paper metrics) · [REPRODUCIBILITY.md](./REPRODUCIBILITY.md) (commands)
