# Data Directory

This directory contains dataset configuration and download scripts for ModernAraBERT.

## Directory Structure

```text
data/
├── README.md          # This file
├── links.json         # Dataset download links configuration
├── benchmarking/      # Task-specific artifacts/datasets for benchmarks (gitignored)
├── raw/               # Raw downloaded datasets (gitignored)
└── preprocessed/      # Intermediate and final processed data for pretraining (gitignored)
```

## Dataset Configuration (`links.json`)

The `links.json` file contains URLs for downloading pretraining corpora from two main sources:

### HuggingFace Datasets

- **Arabic Billion Words**: Large-scale Arabic text corpus from MohamedRashad/arabic-billion-words

### Text Links

- **Wikipedia Multistream XML (txt route)**: 10 Arabic Wikipedia multistream XML `.bz2` parts (May 2025)
  - Keys: `wiki_dumb_txt_1` … `wiki_dumb_txt_10`
  - Note: These URLs point to the same multistream XML `.bz2` files as in `xml_links` and are processed into extracted text during preprocessing.

### XML Links

- **Wikipedia Multistream XML**: Original Arabic Wikipedia multistream XML `.bz2` parts used for XML-based preprocessing
  - Keys: `wiki_dumb_xml_1` … `wiki_dumb_xml_10`
  - These mirror the links under `text_links` in this repository configuration.

## Downloading Datasets

### Pretraining Data

To download all pretraining datasets:

```bash
python scripts/pretraining/run_data_collection.py
```

This will:

1. Read URLs from `links.json`
2. Download Hugging Face datasets and Wikipedia dumps to `data/raw/`
3. Extract compressed files (`.bz2` multistream XML) into `data/raw/extracted/`
4. Log progress to `logs/data_collection.log`

### Benchmarking Data

Benchmark datasets are automatically downloaded through Hugging Face `datasets` library:

- **Sentiment Analysis**: HARD, AJGT, LABR
- **NER**: ANERCorp
- **QA**: Arabic-SQuAD, ARCD

## Data Preprocessing

### Pretraining Corpus Preprocessing

The preprocessing pipeline includes:

1. **Diacritics Removal**: Remove Arabic diacritical marks
2. **Elongation Removal**: Remove tatweel (ـ) characters
3. **Punctuation Cleaning**: Standardize punctuation
4. **Farasa Segmentation**: Apply morphological segmentation
5. **Filtering**: Remove empty lines and very short sentences

Run preprocessing:

```bash
python scripts/pretraining/run_data_preprocessing.py \
  --input-dir data/raw/extracted \
  --output-dir data/preprocessed \
  --all
```

Or run individual stages (examples):

```bash
# 1) Extract text from Wikipedia XML into data/preprocessed/extracted
python scripts/pretraining/run_data_preprocessing.py --input-dir data/raw --output-dir data/preprocessed --process-xml

# 2) Clean/normalize into data/preprocessed/processed
python scripts/pretraining/run_data_preprocessing.py --input-dir data/preprocessed/extracted --output-dir data/preprocessed --process-text

# 3) Segment into data/preprocessed/segmented
python scripts/pretraining/run_data_preprocessing.py --input-dir data/preprocessed/processed --output-dir data/preprocessed --segment

# 4) Create splits under data/preprocessed/splits
python scripts/pretraining/run_data_preprocessing.py --input-dir data/preprocessed/segmented --output-dir data/preprocessed --split
```

### Preprocessed Data Layout

Processed artifacts are organized under `data/preprocessed/`:

- `extracted/`: Text extracted from Wikipedia multistream XML parts
- `segmented/`: Text after morphological segmentation (e.g., Farasa)
- `processed/`: Cleaned/normalized corpus ready for training
- `splits/`: Train/validation/test splits for pretraining

## Adding Custom Datasets

To add a new dataset:

1. **Add URL to `links.json`**:

   ```json
   {
     "huggingface_datasets": {
       "my_hf_dataset": "username/dataset-name"
     },
     "text_links": {
       "my_text_dataset": "https://example.com/dataset.txt.bz2"
     },
     "xml_links": {
       "my_xml_dataset": "https://example.com/dataset.xml.bz2"
     }
   }
   ```

2. **Update data collection script** (if special handling needed)

3. **Add preprocessing** in `src/pretraining/data_preprocessing.py`

## Dataset Statistics

### Pretraining Corpus

| Source                      | Size       | Articles/Sentences | Tokens (approx) |
| --------------------------- | ---------- | ------------------ | --------------- |
| Arabic Billion Words (HF)   | ~8GB       | 3.5M+ articles     | 800M+           |
| Arabic Wikipedia (10 dumps) | ~1.8GB     | 800K+ articles     | 200M+           |
| **Total**                   | **~9.8GB** | **4.3M+**          | **1.0B+**       |

### Benchmark Datasets

| Task | Dataset  | Metric   |
| ---- | -------- | -------- |
| SA   | HARD     | Macro-F1 |
| SA   | AJGT     | Macro-F1 |
| SA   | LABR     | Macro-F1 |
| NER  | ANERCorp | Macro F1 |
| QA   | ARCD     | EM       |


## Contact

For dataset-related questions, please open an issue on GitHub.
