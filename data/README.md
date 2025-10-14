# Data Directory

This directory contains dataset configuration and download scripts for ModernAraBERT.

## Directory Structure

```text
data/
├── README.md          # This file
├── links.json         # Dataset download links configuration
├── raw/               # Raw downloaded datasets (gitignored)
├── interim/           # Intermediate processed data (gitignored)
└── processed/         # Final processed datasets (gitignored)
```

## Dataset Configuration (`links.json`)

The `links.json` file contains URLs for downloading pretraining corpora from two main sources:

### HuggingFace Datasets

- **Arabic Billion Words**: Large-scale Arabic text corpus from MohamedRashad/arabic-billion-words

### Text Links

- **Wikipedia Dumps**: 10 Arabic Wikipedia dumps from wikimedia.org (May 2025)
- **Multistream XML**: Compressed Wikipedia XML dumps for preprocessing

### XML Links

- **Wikipedia XML**: Original Wikipedia XML dumps for preprocessing (same as text_links)

## Downloading Datasets

### Pretraining Data

To download all pretraining datasets:

```bash
python scripts/pretraining/run_data_collection.py
```

This will:

1. Read URLs from `links.json`
2. Download HuggingFace datasets and Wikipedia dumps to `data/raw/`
3. Extract compressed files (.bz2 archives)
4. Log progress to `logs/data_collection.log`

### Benchmarking Data

Benchmark datasets are automatically downloaded through Hugging Face `datasets` library:

- **Sentiment Analysis**: HARD, AJGT, LABR
- **NER**: ANERCorp (via CAMeL Tools)
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
python scripts/pretraining/run_data_collection.py --preprocess-only
```

### Benchmark Data Preprocessing

Each benchmark has task-specific preprocessing:

- **SA**: Text normalization, label encoding
- **NER**: IOB2 tagging, subword alignment
- **QA**: Context-question pair formatting, span extraction

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

| Source | Size | Articles/Sentences | Tokens (approx) |
|--------|------|-------------------|-----------------|
| Arabic Billion Words (HF) | ~8GB | 5M+ articles | 1.5B+ |
| Arabic Wikipedia (10 dumps) | ~15GB | 1M+ articles | 500M+ |
| **Total** | **~23GB** | **6M+** | **2B+** |

### Benchmark Datasets

| Task | Dataset | Train | Val | Test | Metric |
|------|---------|-------|-----|------|--------|
| SA | HARD | 84,585 | - | 9,398 | Macro-F1 |
| SA | AJGT | 1,080 | 360 | 360 | Macro-F1 |
| SA | LABR | 15,000 | 2,500 | 5,000 | Macro-F1 |
| NER | ANERCorp | 11,800 | 1,500 | 1,700 | Micro F1 |
| QA | ARCD | 698 + Arabic-SQuAD | - | 697 | EM/F1/SM |

## Data Licenses

Please refer to the original dataset papers and repositories for licensing information:

- **Arabic Billion Words (HuggingFace)**: See dataset page for license
- **Wikipedia**: CC BY-SA 3.0
- **HARD**: See original paper
- **AJGT**: GitHub repository license
- **LABR**: Research use
- **ANERCorp**: Research use
- **Arabic-SQuAD/ARCD**: Research use

## Storage Requirements

Ensure sufficient disk space:

- Raw data: ~25GB
- Processed data: ~15GB
- Temporary files: ~10GB
- **Total recommended**: 50GB free space

## Troubleshooting

### Download Issues

**Problem**: Download fails or times out

**Solution**:

```bash
# Retry with increased timeout
python scripts/pretraining/run_data_collection.py --timeout 600
```

**Problem**: Google Drive rate limiting

**Solution**: Wait or use alternative mirrors if available

### Preprocessing Issues

**Problem**: Out of memory during preprocessing

**Solution**:

```bash
# Process in smaller chunks
python scripts/pretraining/run_data_collection.py --chunk-size 1000000
```

**Problem**: Farasa segmentation errors

**Solution**: Ensure Farasa is properly installed:

```bash
pip install --upgrade farasa
```

## Citation

If you use these datasets, please cite the original papers:

```bibtex
@article{el20161,
  title={1.5 billion words arabic corpus},
  author={El-Khair, Ibrahim Abu},
  journal={arXiv preprint arXiv:1611.04033},
  year={2016}
}

@misc{wikipedia2025,
  title={Arabic Wikipedia Dumps},
  author={Wikimedia Foundation},
  year={2025},
  url={https://dumps.wikimedia.org/arwiki/}
}

@misc{mohamedrashad2024,
  title={Arabic Billion Words Dataset},
  author={Mohamed Rashad},
  year={2024},
  url={https://huggingface.co/datasets/MohamedRashad/arabic-billion-words}
}

% Add other dataset citations as needed
```

## Contact

For dataset-related questions:

- Open an issue on GitHub
- Contact: [ahmed.aldamati@gizasystems.com](mailto:ahmed.aldamati@gizasystems.com)
