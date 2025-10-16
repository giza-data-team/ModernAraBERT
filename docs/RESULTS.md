# Results Summary

This page reflects the results reported in the paper. For exact reproduction commands, see [REPRODUCIBILITY.md](./REPRODUCIBILITY.md).

---

## Sentiment Analysis — Macro-F1

| Model         | LABR  | HARD  | AJGT  |
| ------------- | ----- | ----- | ----- |
| AraBERTv1     | 45.35 | 72.65 | 58.01 |
| AraBERTv2     | 45.79 | 67.10 | 53.59 |
| mBERT         | 44.18 | 71.70 | 61.55 |
| MARBERT       | 45.54 | 67.39 | 60.63 |
| ModernAraBERT | 56.45 | 89.37 | 70.54 |

---

## Named Entity Recognition — Macro-F1

| Model         | NER (Macro-F1) |
| ------------- | -------------- |
| AraBERTv1     | 13.46          |
| AraBERTv2     | 16.77          |
| mBERT         | 12.15          |
| MARBERT       | 7.42           |
| ModernAraBERT | 28.23          |

---

## Question Answering (ARCD Test) — Exact Match (EM, %)

| Model         | EM    |
| ------------- | ----- |
| AraBERT       | 25.36 |
| AraBERTv2     | 26.08 |
| mBERT         | 25.12 |
| MARBERT       | 23.58 |
| ModernAraBERT | 27.10 |

---

## Variance and Reproducibility

- Results were obtained with fixed seed 42. Minor variations may occur with different hardware/software stacks.
- See [REPRODUCIBILITY.md](./REPRODUCIBILITY.md) for commands, seeds, and environment notes.
