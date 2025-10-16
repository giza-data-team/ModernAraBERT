# Datasets

Sources and settings used in ModernAraBERT. For detailed reproduction and counts, see [REPRODUCIBILITY.md](./REPRODUCIBILITY.md).

## Pretraining corpora (~17GB preprocessed)
- OSIAN — News (MSA)
- Arabic Billion Words — Mixed web/news
- Arabic Wikipedia — Encyclopedia (MSA)
- OSCAR Arabic — Web-crawled (MSA + dialect)

Pipeline (download + preprocess):
```bash
python scripts/pretraining/run_data_collection.py --config configs/pretraining/data_collection.yaml
```
Tokenizer extension is optional but recommended (see [PRETRAINING.md](./PRETRAINING.md)).

## Benchmarks
- Sentiment (Macro-F1): HARD (binary, no 3-star), AJGT (1,800 tweets), LABR (binary, std split)
- NER (Macro-F1): ANERCorp (official CAMeL splits)
- QA (EM on ARCD test): Train on Arabic-SQuAD + 50% ARCD; test on remaining 50% ARCD

## Versioning (reproducibility)
- AJGT: GitHub `komari6/Arabic-twitter-corpus-AJGT` (snapshot 2024-12)
- HARD: HF `Elnagara/hard` (binary filtered per AraBERT)
- LABR: ACL 2013 binary split
- ANERCorp: HF `asas-ai/ANERCorp`
- ARCD/Arabic-SQuAD: per original papers

## Quick access via HF datasets
```python
from datasets import load_dataset
load_dataset("Elnagara/hard")
load_dataset("asas-ai/ANERCorp")
load_dataset("mohamedadaly/labr")
```

Links: [RESULTS.md](./RESULTS.md) (paper metrics) · [REPRODUCIBILITY.md](./REPRODUCIBILITY.md) (commands)
