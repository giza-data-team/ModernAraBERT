# Model Card: ModernAraBERT (Concise)

## Summary
- Arabic encoder adapted from `answerdotai/ModernBERT-base` via continued pretraining on Arabic corpora (~9.8GB).
- Strong results across SA, NER (Macro-F1), and QA EM vs. AraBERT/mBERT/MARBERT.
- License: MIT · Paper: LREC 2026 · Hub: gizadatateam/ModernAraBERT

Links: `docs/RESULTS.md` (metrics) · `docs/REPRODUCIBILITY.md` (exact commands)

## Intended Uses
- Masked LM, feature extraction, and transfer learning for Arabic tasks.
- Downstream: sentiment analysis, NER, extractive QA, general classification/labeling.

Out of scope: MT, open-ended generation, heavy code-switching.

## How to use
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
name = "gizadatateam/ModernAraBERT"
model = AutoModelForMaskedLM.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)
```

## Training data and recipe (brief)
- Corpora: OSIAN, Arabic Billion Words, Arabic Wikipedia, OSCAR Arabic
- Tokenizer: ModernBERT vocab + 80K Arabic tokens
- Objective: MLM (3 epochs; 128→512 seq len)
- Hardware: A100 40GB; framework: PyTorch + Transformers + Accelerate

## Evaluation (from paper)

### Sentiment Analysis — Macro-F1 (%)
| Model             | LABR      | HARD      | AJGT      |
| ----------------- | --------- | --------- | --------- |
| AraBERTv1         | 45.35     | 72.65     | 58.01     |
| AraBERTv2         | 45.79     | 67.10     | 53.59     |
| mBERT             | 44.18     | 71.70     | 61.55     |
| MARBERT           | 45.54     | 67.39     | 60.63     |
| **ModernAraBERT** | **56.45** | **89.37** | **70.54** |

### NER — Macro-F1 (%)
| Model             | Macro-F1  |
| ----------------- | --------- |
| AraBERTv1         | 13.46     |
| AraBERTv2         | 16.77     |
| mBERT             | 12.15     |
| MARBERT           | 7.42      |
| **ModernAraBERT** | **28.23** |

### QA (ARCD test) — EM (%)
| Model             | EM        |
| ----------------- | --------- |
| AraBERT           | 25.36     |
| AraBERTv2         | 26.08     |
| mBERT             | 25.12     |
| MARBERT           | 23.58     |
| **ModernAraBERT** | **27.10** |

## Risks & limitations (brief)
- Primarily MSA; some dialectal coverage.
- Higher VRAM vs. AraBERT; see trade-offs in paper.
- Potential biases inherited from web/news sources.

## Citation
```bibtex
@inproceedings{<paper_id>,
  title={Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic},
  author={Maher, Eldamaty, Ashraf, ElShawi, Mostafa},
  booktitle={Proceedings of <conference_name>},
  year={2025},
  organization={<conference_name>}
}
```
