# Datasets Documentation

Complete documentation of all datasets used for pretraining and benchmarking ModernAraBERT.

## Table of Contents

- [Pretraining Corpora](#pretraining-corpora)
- [Benchmarking Datasets](#benchmarking-datasets)
- [Data Preprocessing](#data-preprocessing)
- [Download Instructions](#download-instructions)
- [Licensing](#licensing)

---

## Pretraining Corpora

ModernAraBERT was pretrained on approximately **17GB** of Arabic text from four publicly available sources.

### 1. OSIAN (Open Source International Arabic News)

**Source**: Zeroual et al. (2019)

- **Size**: ~2GB
- **Content**: International Arabic news articles
- **Language**: Modern Standard Arabic (MSA)
- **Sentences**: ~1.5M
- **Domain**: News, journalism

**Citation**:
```bibtex
@inproceedings{zeroual-etal-2019-osian,
  title={OSIAN: Open Source International Arabic News Corpus},
  author={Zeroual, Imad and Goldhahn, Dirk and Eckart, Thomas and Lakhouaja, Abdelhak},
  booktitle={Proceedings of WANLP},
  year={2019}
}
```

### 2. Arabic Billion Words Corpus

**Source**: El-Khair (2016)

- **Size**: ~8GB
- **Content**: Diverse web text, news, and articles
- **Language**: MSA and some dialectal
- **Tokens**: ~800M
- **Domain**: Mixed (news, web, forums)

**Citation**:
```bibtex
@article{el20161,
  title={1.5 billion words arabic corpus},
  author={El-Khair, Ibrahim Abu},
  journal={arXiv preprint arXiv:1611.04033},
  year={2016}
}
```

### 3. Arabic Wikipedia

**Source**: Wikimedia Foundation

- **Size**: ~3GB
- **Content**: Encyclopedia articles
- **Language**: Modern Standard Arabic
- **Sentences**: ~800K
- **Domain**: Encyclopedia, factual content
- **Dump Date**: January 2025
- **URL**: https://dumps.wikimedia.org/arwiki/

**License**: CC BY-SA 3.0

### 4. OSCAR Arabic

**Source**: OSCAR 2022 (Abadji et al., 2022)

- **Size**: ~4GB
- **Content**: Web-crawled Arabic text
- **Language**: MSA and dialectal mix
- **Sentences**: ~700K
- **Domain**: Web pages, diverse
- **URL**: https://oscar-project.org/

**Citation**:
```bibtex
@article{2022arXiv220106642A,
  author = {Abadji, Julien and Ortiz Suarez, Pedro and Romary, Laurent and Sagot, Beno{\^i}t},
  title = {Towards a Cleaner Document-Oriented Multilingual Crawled Corpus},
  journal = {arXiv e-prints},
  year = 2022
}
```

**License**: CC0 1.0

### Total Pretraining Corpus

| Source | Size | Sentences | Tokens (approx) |
|--------|------|-----------|-----------------|
| OSIAN | 2GB | 1.5M | 350M |
| Arabic Billion Words | 8GB | 3.5M | 800M |
| Arabic Wikipedia | 3GB | 800K | 200M |
| OSCAR Arabic | 4GB | 700K | 150M |
| **Total** | **17GB** | **6.5M** | **1.5B** |

---

## Benchmarking Datasets

### Sentiment Analysis

#### 1. HARD (Hotel Arabic Reviews Dataset)

**Source**: Elnagar et al. (2018)

- **Task**: Binary sentiment classification
- **Size**: 93,983 reviews (after filtering)
  - Train: 84,585
  - Test: 9,398
- **Classes**: Positive, Negative (neutral excluded)
- **Language**: MSA + Dialectal mix
- **Domain**: Hotel reviews

**Special Notes**:
- Excludes 3-star (neutral) reviews as per AraBERT paper
- Mix of Modern Standard Arabic and dialectal forms

**Citation**:
```bibtex
@Inbook{Elnagar2018,
  author="Elnagar, Ashraf and Khalifa, Yasmin S. and Einea, Anas",
  title="Hotel Arabic-Reviews Dataset Construction for Sentiment Analysis",
  booktitle="Intelligent Natural Language Processing",
  year="2018",
  publisher="Springer"
}
```

#### 2. AJGT (Arabic Jordanian General Tweets)

**Source**: GitHub repository

- **Task**: Binary sentiment classification
- **Size**: 1,800 labeled tweets
  - Train: 1,080 (60%)
  - Validation: 360 (20%)
  - Test: 360 (20%)
- **Classes**: Positive, Negative
- **Language**: Jordanian dialect
- **Domain**: Social media (Twitter)

**URL**: https://github.com/komari6/Arabic-twitter-corpus-AJGT

#### 3. LABR (Large-Scale Arabic Book Reviews)

**Source**: Aly & Atiya (2013)

- **Task**: Binary sentiment classification
- **Size**: 22,500 book reviews (unbalanced binary version)
  - Train: 15,000
  - Validation: 2,500
  - Test: 5,000
- **Classes**: Positive, Negative
- **Language**: Modern Standard Arabic
- **Domain**: Book reviews

**Citation**:
```bibtex
@inproceedings{aly-atiya-2013-labr,
  title={LABR: A Large Scale Arabic Book Reviews Dataset},
  author={Aly, Mohamed and Atiya, Amir},
  booktitle={ACL},
  year={2013}
}
```

### Named Entity Recognition

#### ANERCorp (Arabic Named Entity Recognition Corpus)

**Source**: Benajiba et al. (2007), distributed via CAMeL Lab

- **Task**: Named Entity Recognition
- **Entity Types**: Person (PER), Location (LOC), Organization (ORG), Miscellaneous (MISC)
- **Tagging Scheme**: IOB2 format
- **Size**:
  - Train: ~11,800 sentences
  - Validation: ~1,500 sentences
  - Test: ~1,700 sentences
- **Language**: Modern Standard Arabic
- **Domain**: News articles

**Access**: Via Hugging Face Datasets
```python
from datasets import load_dataset
dataset = load_dataset("CAMeL-Lab/ANERcorp")
```

**Citation**:
```bibtex
@inproceedings{Benajiba:2007,
  title={ANERsys: An Arabic Named Entity Recognition System},
  author={Benajiba, Yassine and Rosso, Paolo and Bened{\'i}Ruiz, Jos{\'e} Miguel},
  booktitle={CICLing},
  year={2007}
}
```

### Question Answering

#### Arabic-SQuAD

**Source**: Mozannar et al. (2019)

- **Task**: Extractive question answering
- **Format**: SQuAD-style (context, question, answer span)
- **Size**: Used for training (50% in our experiments)
- **Language**: Modern Standard Arabic
- **Domain**: Wikipedia

**Citation**:
```bibtex
@article{Mozannar:2019,
  title={Neural Arabic Question Answering},
  author={Mozannar, Hussein and others},
  journal={arXiv preprint arXiv:1906.05685},
  year={2019}
}
```

#### ARCD (Arabic Reading Comprehension Dataset)

**Source**: Mozannar et al. (2019)

- **Task**: Extractive question answering
- **Format**: SQuAD-style
- **Size**: 1,395 questions
  - Train: 50% + Arabic-SQuAD (698 questions)
  - Test: 50% (697 questions)
- **Language**: Modern Standard Arabic
- **Domain**: Wikipedia

**Metrics**:
- **EM (Exact Match)**: Exact string match percentage
- **F1**: Token-level F1 score
- **SM (Sentence Match)**: Sentence-level semantic match

**Same citation as Arabic-SQuAD**

---

## Data Preprocessing

### Pretraining Corpus Preprocessing

All pretraining corpora undergo the following pipeline:

1. **Diacritics Removal**
   ```python
   import re
   text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove tashkeel
   ```

2. **Elongation Removal**
   ```python
   text = text.replace('ـ', '')  # Remove tatweel
   ```

3. **Punctuation Normalization**
   - Standardize Arabic and English punctuation
   - Remove excessive punctuation

4. **Farasa Segmentation** (Optional but recommended)
   ```python
   from farasa.segmenter import FarasaSegmenter
   segmenter = FarasaSegmenter()
   text = segmenter.segment(text)
   ```

5. **Filtering**
   - Remove empty lines
   - Remove sentences < 5 words
   - Remove sentences > 512 tokens (after tokenization)

### Benchmark Dataset Preprocessing

#### Sentiment Analysis
- Text normalization (same as pretraining)
- Label encoding (positive=1, negative=0)
- Truncation to 512 tokens

#### Named Entity Recognition
- IOB2 tagging
- First-subtoken labeling for BERT tokenization
- Continuation tokens masked with -100 (ignored in loss)

#### Question Answering
- Context-question pair formatting
- Maximum length: 512 tokens
- Document stride: 128 tokens (for long contexts)
- Character-to-token span mapping

---

## Download Instructions

### Automatic Download

Use our data collection script:

```bash
python scripts/pretraining/run_data_collection.py \
    --config configs/pretraining_config.yaml \
    --download-only
```

This downloads all pretraining corpora to `data/raw/`.

### Manual Download

#### Wikipedia Dumps

```bash
# Download latest Arabic Wikipedia dump
wget https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2

# Extract
bzip2 -d arwiki-latest-pages-articles.xml.bz2
```

#### OSCAR Arabic

Visit https://oscar-project.org/ and download the Arabic subset.

#### Google Drive Files

For One Billion Words corpus parts hosted on Google Drive, use `gdown`:

```bash
pip install gdown

# Example for one file
gdown https://drive.google.com/uc?id=FILE_ID
```

### Benchmark Datasets

Most benchmark datasets are automatically downloaded via Hugging Face `datasets`:

```python
from datasets import load_dataset

# HARD
dataset = load_dataset("Elnagara/hard", "plain_text")

# ANERCorp
dataset = load_dataset("CAMeL-Lab/ANERcorp")

# LABR
dataset = load_dataset("labr")
```

**AJGT** requires manual download from GitHub:
```bash
git clone https://github.com/komari6/Arabic-twitter-corpus-AJGT.git
```

---

## Licensing

### Pretraining Corpora

| Dataset | License | Commercial Use |
|---------|---------|----------------|
| OSIAN | Research use | Check with authors |
| Arabic Billion Words | CC BY-NC 4.0 | Non-commercial only |
| Wikipedia | CC BY-SA 3.0 | ✓ Yes |
| OSCAR | CC0 1.0 | ✓ Yes |

### Benchmark Datasets

| Dataset | License | Commercial Use |
|---------|---------|----------------|
| HARD | Research use | Check with authors |
| AJGT | GitHub repo license | Check repo |
| LABR | Research use | Check with authors |
| ANERCorp | Research use | Check with authors |
| Arabic-SQuAD | Research use | Check with authors |
| ARCD | Research use | Check with authors |

**Important**: Always cite the original dataset papers and respect their licenses.

---

## Dataset Statistics Summary

### Pretraining

- **Total Size**: 17GB (preprocessed)
- **Total Sentences**: 6.5M+
- **Total Tokens**: 1.5B+
- **Languages**: MSA (majority) + dialectal varieties
- **Domains**: News, web, encyclopedia

### Benchmarking

- **Total Tasks**: 3 (SA, NER, QA)
- **Total Datasets**: 6 (HARD, AJGT, LABR, ANERCorp, Arabic-SQuAD, ARCD)
- **Total Samples**: ~120K (combined across all tasks)

---

## Data Quality

### Quality Control Measures

1. **Duplicate Removal**: Deduplicated at sentence level
2. **Language Detection**: Filtered non-Arabic content
3. **Length Filtering**: Removed very short/long sequences
4. **Encoding**: UTF-8 validated
5. **Formatting**: Consistent text formatting across sources

### Known Issues

- **Dialectal Mix**: Some datasets mix MSA and dialects
- **OCR Errors**: Some sources may contain OCR errors
- **Code-switching**: Occasional English/Arabic code-mixing
- **Noise**: Web-scraped data may contain HTML artifacts

---

## Contact

For dataset-related questions:
- GitHub Issues: [ModernAraBERT/issues](https://github.com/giza-data-team/ModernAraBERT/issues)
- Email: ahmed.aldamati@gizasystems.com

---

## Citation

If you use these datasets, please cite the original papers and our work:

```bibtex
@inproceedings{eldamaty2026modernarabert,
  title={Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic},
  author={Eldamaty, Ahmed and Maher, Mohamed and Mostafa, Mohamed and Ashraf, Mariam and ElShawi, Radwa},
  booktitle={Proceedings of LREC-COLING 2026},
  year={2026}
}
```

