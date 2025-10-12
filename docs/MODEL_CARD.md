# Model Card for ModernAraBERT

## Model Details

### Model Description

ModernAraBERT is a state-of-the-art Arabic language model created through efficient adaptation of the English-pretrained [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base). Our approach leverages continued pretraining on large-scale Arabic corpora, preserving cross-lingual knowledge while capturing Arabic-specific morphological features.

- **Developed by:** Giza Systems and University of Tartu
- **Model type:** Masked Language Model (BERT-style encoder)
- **Language(s):** Arabic (Modern Standard Arabic and dialects)
- **License:** MIT
- **Base Model:** [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
- **Paper:** [Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic](https://github.com/giza-data-team/ModernAraBERT) (LREC 2026)
- **Repository:** [GitHub](https://github.com/giza-data-team/ModernAraBERT)

### Model Sources

- **Repository:** https://github.com/giza-data-team/ModernAraBERT
- **Paper:** LREC-COLING 2026
- **Hugging Face:** https://huggingface.co/gizadatateam/ModernAraBERT

## Uses

### Direct Use

ModernAraBERT can be used for:

- **Masked Language Modeling**: Fill-in-the-blank tasks for Arabic text
- **Feature Extraction**: Generate contextual embeddings for Arabic text
- **Transfer Learning**: Fine-tune on downstream Arabic NLP tasks

### Downstream Use

The model excels at:

1. **Sentiment Analysis**: Classifying sentiment in Arabic text
   - Social media posts
   - Product/service reviews
   - News articles

2. **Named Entity Recognition**: Identifying persons, locations, organizations in Arabic
   - Information extraction
   - Document processing
   - Content categorization

3. **Question Answering**: Extracting answers from Arabic contexts
   - Search systems
   - Chatbots
   - Educational applications

4. **Text Classification**: General Arabic text categorization
5. **Sequence Labeling**: Part-of-speech tagging, chunking
6. **Semantic Similarity**: Comparing Arabic text similarity

### Out-of-Scope Use

- **Machine Translation**: Not optimized for translation tasks (use encoder-decoder models)
- **Text Generation**: Not designed for generative tasks (use GPT-style models)
- **Low-resource Arabic dialects**: Primarily trained on MSA with some dialectal coverage
- **Code-switching**: Limited support for Arabic-English code-mixed text

## Bias, Risks, and Limitations

### Known Limitations

1. **Dialectal Coverage**: While including some dialectal Arabic, the model is primarily optimized for Modern Standard Arabic (MSA)

2. **Domain Bias**: Training data includes news articles, Wikipedia, and web text, which may not generalize perfectly to specialized domains (e.g., medical, legal)

3. **Computational Requirements**: Higher GPU memory usage compared to AraBERT (see resource analysis in paper)

4. **Cultural Context**: May not capture all cultural nuances specific to different Arabic-speaking regions

### Potential Biases

- **Geographic Bias**: Training data may overrepresent certain Arabic-speaking regions
- **Temporal Bias**: Reflects language and cultural norms from the training data collection period
- **Socioeconomic Bias**: Web and news sources may underrepresent certain socioeconomic groups
- **Gender Bias**: Inherited biases from both the English-pretrained ModernBERT and Arabic corpora

### Recommendations

Users should:
- Evaluate the model on their specific use case before deployment
- Consider potential biases when using for sensitive applications
- Fine-tune on domain-specific data for specialized applications
- Implement appropriate safeguards for production systems
- Monitor model outputs for unexpected behaviors

## How to Get Started with the Model

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load model and tokenizer
model_name = "gizadatateam/ModernAraBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: Masked Language Modeling
text = "القاهرة هي [MASK] مصر"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# Get top predictions for masked token
masked_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"Predicted word: {predicted_token}")  # Expected: "عاصمة" (capital)
```

### Fine-tuning for Sentiment Analysis

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    "gizadatateam/ModernAraBERT",
    num_labels=2  # Binary sentiment
)
tokenizer = AutoTokenizer.from_pretrained("gizadatateam/ModernAraBERT")

# Freeze encoder (train only classification head)
for param in model.base_model.parameters():
    param.requires_grad = False

# Prepare your dataset
# train_dataset = ...

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=50,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

### Feature Extraction

```python
from transformers import AutoTokenizer, AutoModel
import torch

model = AutoModel.from_pretrained("gizadatateam/ModernAraBERT")
tokenizer = AutoTokenizer.from_pretrained("gizadatateam/ModernAraBERT")

text = "مرحبا بك في عالم البرمجة"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    
# Get embeddings
embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
```

## Training Details

### Training Data

ModernAraBERT was trained on ~17GB of Arabic text from four sources:

1. **OSIAN** (Open Source International Arabic News Corpus)
   - Size: ~2GB
   - Content: News articles in MSA

2. **Arabic Billion Words Corpus**
   - Size: ~8GB
   - Content: Diverse web text

3. **Arabic Wikipedia**
   - Size: ~3GB
   - Content: Encyclopedia articles in MSA

4. **OSCAR Arabic**
   - Size: ~4GB
   - Content: Web-crawled Arabic text

**Total**: 6M+ sentences, 1.5B+ tokens

### Training Procedure

#### Preprocessing

1. **Diacritics Removal**: Removed Arabic diacritical marks (tashkeel)
2. **Elongation Removal**: Removed tatweel (ـ) characters
3. **Punctuation Standardization**: Normalized punctuation
4. **Farasa Segmentation**: Applied morphological segmentation

#### Tokenization

- **Base Vocabulary**: ModernBERT vocabulary (50,368 tokens)
- **Extended Vocabulary**: +80,000 Arabic-specific tokens
- **Total Vocabulary**: 130,368 tokens
- **Tokenizer**: Modified ModernBERT WordPiece tokenizer

#### Training Hyperparameters

- **Objective**: Masked Language Modeling (MLM) with 15% masking
- **Base Model**: ModernBERT-base (22 layers, 768 hidden size)
- **Epochs**: 3
  - Epochs 1-2: 128 max sequence length
  - Epoch 3: 512 max sequence length
- **Batch Size**: 32 per GPU
- **Learning Rate**: 1×10⁻⁴
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95, weight decay=0.01)
- **LR Schedule**: Cosine with warmup
- **Warmup Steps**: 10,000
- **Gradient Clipping**: Max norm = 1.0
- **Mixed Precision**: FP16

#### Training Infrastructure

- **Hardware**: NVIDIA A100 40GB GPU
- **CPU**: 12 cores
- **RAM**: 32GB
- **Training Time**: ~60 hours total
- **Framework**: PyTorch + Hugging Face Transformers + Accelerate

#### Speeds, Sizes, Times

- **Model Size**: ~570MB (model weights)
- **Vocabulary Size**: 130,368 tokens
- **Parameters**: ~149M (base model + extended embeddings)
- **Training Duration**: 60 hours on A100
- **Inference Speed**: ~500 samples/sec (NER task, A100)

## Evaluation

### Testing Data, Factors & Metrics

#### Datasets

1. **Sentiment Analysis**
   - HARD (Hotel Arabic Reviews)
   - AJGT (Arabic Jordanian General Tweets)
   - LABR (Large Arabic Book Reviews)
   - **Metric**: Macro-F1

2. **Named Entity Recognition**
   - ANERCorp
   - **Metric**: Micro F1

3. **Question Answering**
   - ARCD (Arabic Reading Comprehension Dataset)
   - **Metrics**: Exact Match (EM), F1-Score, Sentence Match (SM)

### Results

#### Sentiment Analysis (Macro-F1 %)

| Dataset | AraBERT v1 | mBERT | ModernAraBERT | Improvement |
|---------|------------|-------|---------------|-------------|
| AJGT | 58.0 | 61.5 | **70.5** | +12.5% |
| HARD | 72.7 | 71.7 | **89.4** | +16.7% |
| LABR | 45.5 | 45.5 | **56.5** | +11.0% |

#### Named Entity Recognition (Micro F1 %)

| Model | Micro F1 |
|-------|----------|
| AraBERT v1 | 78.9 |
| ModernAraBERT | 82.1 |
| mBERT | **90.7** |

#### Question Answering (%)

| Metric | AraBERT v1 | mBERT | ModernAraBERT | Improvement |
|--------|------------|-------|---------------|-------------|
| Exact Match | 13.26 | 15.27 | **18.73** | +41.3% |
| F1-Score | 40.82 | 46.12 | **47.18** | +15.6% |
| Sentence Match | 71.47 | 63.11 | **76.66** | +7.3% |

### Summary

- **Strengths**: Excels at sentence-level tasks (SA, QA), strong semantic understanding
- **Competitive**: Achieves competitive or superior performance vs. established baselines
- **Trade-offs**: Higher VRAM usage but better accuracy compared to AraBERT

## Environmental Impact

- **Hardware Type**: NVIDIA A100 40GB
- **Hours Used**: ~60 hours
- **Cloud Provider**: [Academic/Research Institution]
- **Carbon Emitted**: Estimated ~12 kg CO₂eq (using ML CO₂ calculator)

*Note: This is an adaptation strategy, not training from scratch, significantly reducing environmental impact compared to full pretraining.*

## Technical Specifications

### Model Architecture

- **Architecture**: Encoder-only transformer (BERT-style)
- **Layers**: 22
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Intermediate Size**: 3072
- **Max Position Embeddings**: 8192 (architecture), 512 (training)
- **Position Embeddings**: Rotary Position Embeddings (RoPE)
- **Attention**: Alternating global-local attention
- **Activation**: GeGLU
- **Dropout**: 0.1
- **Layer Norm**: Pre-LayerNorm

### Compute Infrastructure

- **GPU**: NVIDIA A100 (40GB)
- **CPU**: 12 cores
- **RAM**: 32GB
- **Storage**: SSD recommended
- **Framework**: PyTorch 2.0+, Transformers 4.35+

## Citation

**BibTeX:**

```bibtex
@inproceedings{eldamaty2026modernarabert,
  title={Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic},
  author={Eldamaty, Ahmed and Maher, Mohamed and Mostafa, Mohamed and Ashraf, Mariam and ElShawi, Radwa},
  booktitle={Proceedings of the 2026 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2026)},
  year={2026},
  organization={ELRA}
}
```

**APA:**

Eldamaty, A., Maher, M., Mostafa, M., Ashraf, M., & ElShawi, R. (2026). Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic. In *Proceedings of LREC-COLING 2026*.

## Model Card Authors

Ahmed Eldamaty, Mohamed Maher, Mohamed Mostafa, Mariam Ashraf, Radwa ElShawi

## Model Card Contact

- **Email**: ahmed.aldamati@gizasystems.com, mohamed.abdelrahman@ut.ee
- **GitHub**: https://github.com/giza-data-team/ModernAraBERT/issues
- **Organization**: Giza Systems & University of Tartu

## Additional Information

### Related Models

- [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base): Base English model
- [AraBERT](https://huggingface.co/aubmindlab/bert-base-arabertv2): Arabic BERT baseline
- [mBERT](https://huggingface.co/bert-base-multilingual-cased): Multilingual BERT

### Future Work

- Extend to more Arabic dialects
- Support for longer contexts (up to 8192 tokens)
- Parameter-efficient fine-tuning techniques
- Multi-task learning across Arabic NLP tasks

