# Results Summary (Paper Alignment)

This page reflects the results reported in the paper. For exact reproduction commands, see [REPRODUCIBILITY.md](./REPRODUCIBILITY.md).

---

## Sentiment Analysis — Macro-F1 (seed=42)

| Model         | LABR  | HARD  | AJGT  |
| ------------- | ----- | ----- | ----- |
| AraBERTv1     | 45.35 | 72.65 | 58.01 |
| AraBERTv2     | 45.79 | 67.10 | 53.59 |
| mBERT         | 44.18 | 71.70 | 61.55 |
| MARBERT       | 45.54 | 67.39 | 60.63 |
| ModernAraBERT | 56.45 | 89.37 | 70.54 |

Notes:

- AJGT requires manual download; see [DATASETS.md](./DATASETS.md).
- HARD excludes 3-star (neutral) reviews (AraBERT convention).

---

## Named Entity Recognition — Macro-F1 (seed=42)

| Model         | NER (Macro-F1) |
| ------------- | -------------- |
| AraBERTv1     | 13.46          |
| AraBERTv2     | 16.77          |
| mBERT         | 12.15          |
| MARBERT       | 7.42           |
| ModernAraBERT | 28.23          |

Note: Paper reports entity-level Macro-F1 (not token-level Micro F1).

---

## Question Answering (ARCD Test) — Exact Match (EM, %) (seed=42)

| Model         | EM    |
| ------------- | ----- |
| AraBERT       | 25.36 |
| AraBERTv2     | 26.08 |
| mBERT         | 25.12 |
| MARBERT       | 23.58 |
| ModernAraBERT | 27.10 |

Note: Paper reports EM only for ARCD in the main table.

---

## Variance and Reproducibility

- Results were obtained with fixed seed 42. Minor variations may occur with different hardware/software stacks.
- See [REPRODUCIBILITY.md](./REPRODUCIBILITY.md) for commands, seeds, and environment notes.

## Dataset Versioning

- AJGT: GitHub `komari6/Arabic-twitter-corpus-AJGT` (snapshot 2024-12)
- HARD: Hugging Face `Elnagara/hard` (binary filtered per AraBERT)
- LABR: ACL 2013 binary version (standard split)
- ANERCorp: Hugging Face `asas-ai/ANERCorp`
- ARCD/Arabic-SQuAD: per original papers; splits as described in [DATASETS.md](./DATASETS.md)
