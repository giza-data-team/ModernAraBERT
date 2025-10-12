# Original Code - Unmodified Implementation

This directory contains the **original, unrefactored code** from the ModernAraBERT project. This code is preserved exactly as it was before the repository restructuring to maintain a reference to the working implementation.

## ⚠️ Important

**Do not modify any files in this directory.** This is the reference implementation.

For the refactored, modular code, see the parent directory structure:
- `../src/` - Refactored source code modules
- `../scripts/` - User-facing executable scripts
- `../docs/` - Documentation

## Contents

```
original_code/
├── benchmarking/
│   ├── ner/
│   │   └── ner_benchmarking-mbert-samy.py
│   └── sa/
│       ├── sa_benchmarking.py
│       ├── text_preprocessing.py
│       └── train.py
├── pretraining/
│   ├── Data collection and preprocessing.py
│   ├── ModernBERT Training.py
│   ├── Tokenizer vocab extending.py
│   └── links.json
├── LREC2026 Author's kit/
│   ├── lrec2026-example.tex
│   ├── lrec2026-example.bib
│   ├── experiments/
│   │   ├── results.tex
│   │   └── setup.tex
│   └── ...
├── ModernAraBert_LREC.pdf
└── README.md (this file)
```

## How the Code Works

### Pretraining Pipeline

1. **Data Collection**: `Data collection and preprocessing.py`
   - Downloads datasets from Google Drive and URLs
   - Extracts compressed archives
   - Logs progress to `pipeline.log`

2. **Preprocessing**: Same file
   - Removes diacritics
   - Applies Farasa segmentation
   - Normalizes text
   - Splits into train/val/test

3. **Tokenizer Extension**: `Tokenizer vocab extending.py`
   - Extends ModernBERT vocabulary with 80K Arabic tokens
   - Analyzes token frequency
   - Trains extended tokenizer

4. **Training**: `ModernBERT Training.py`
   - Continued pretraining with MLM objective
   - Multi-stage sequence lengths (128 → 512)
   - Checkpoint saving and logging

### Benchmarking

#### Sentiment Analysis (`benchmarking/sa/`)
- `sa_benchmarking.py`: Main benchmarking script
- `text_preprocessing.py`: Arabic text preprocessing utilities
- `train.py`: Training loop with early stopping

**Datasets**: HARD, AJGT, LABR

#### Named Entity Recognition (`benchmarking/ner/`)
- `ner_benchmarking-mbert-samy.py`: Complete NER benchmarking
- Uses first-subtoken labeling strategy
- Macro-F1 evaluation
- Memory and throughput tracking

**Dataset**: ANERCorp

## Running the Original Code

### Prerequisites

```bash
# Install dependencies
pip install torch transformers datasets accelerate farasa sklearn pandas numpy tqdm psutil gdown rarfile
```

### Pretraining

```bash
# 1. Download and preprocess data
python pretraining/"Data collection and preprocessing.py"

# 2. Extend tokenizer
python pretraining/"Tokenizer vocab extending.py"

# 3. Train model
python pretraining/"ModernBERT Training.py"
```

### Benchmarking

```bash
# Sentiment Analysis
python benchmarking/sa/sa_benchmarking.py --model-name modernbert --dataset hard

# Named Entity Recognition
python benchmarking/ner/ner_benchmarking-mbert-samy.py --model-name modernbert
```

## Differences from Refactored Code

The refactored code (`../src/`) differs in structure but maintains identical logic:

| Original | Refactored | Changes |
|----------|------------|---------|
| Single large files | Modular structure | Code split into focused modules |
| Hardcoded paths | Config files | Externalized configuration |
| Mixed concerns | Separated concerns | Data/training/evaluation split |
| Direct execution | Scripts + imports | Wrapper scripts call modules |
| Inline documentation | Comprehensive docs | Separate documentation files |

**Core logic remains unchanged** - only organization and structure were modified.

## Why Preserve This?

1. **Reference**: Verify refactored code maintains identical behavior
2. **Comparison**: Understand what changed during restructuring
3. **Debugging**: Fall back to working implementation if issues arise
4. **Validation**: Ensure no logic was accidentally modified

## Paper Source

The `LREC2026 Author's kit/` directory contains the LaTeX source for the paper:
- **Title**: "Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic"
- **Authors**: Ahmed Eldamaty, Mohamed Maher, Mohamed Mostafa, Mariam Ashraf, Radwa ElShawi
- **Conference**: LREC-COLING 2026

## License

Same as parent project: MIT License

## Contact

For questions about the original implementation:
- Ahmed Eldamaty: ahmed.aldamati@gizasystems.com
- Mohamed Maher: mohamed.abdelrahman@ut.ee

---

**Remember**: This is a read-only reference. All development should happen in the refactored code structure in the parent directory.

