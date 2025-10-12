# ModernAraBERT Pretraining Guide

This guide provides comprehensive instructions for pretraining ModernAraBERT from scratch using the continued pretraining strategy described in our paper.

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Data Preparation](#data-preparation)
- [Tokenizer Extension](#tokenizer-extension)
- [Training Process](#training-process)
- [Monitoring and Evaluation](#monitoring-and-evaluation)
- [Troubleshooting](#troubleshooting)

---

## Overview

ModernAraBERT is created through **continued pretraining** of the English-pretrained [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) model on Arabic corpora. This approach:

- ✅ Preserves cross-lingual knowledge from English pretraining
- ✅ Adapts to Arabic morphology through continued training
- ✅ More resource-efficient than training from scratch
- ✅ Achieves superior performance on Arabic NLP tasks

### Pretraining Strategy

1. **Start with ModernBERT-base** (English-pretrained)
2. **Extend vocabulary** with 80K Arabic tokens
3. **Continue pretraining** on Arabic corpora using MLM objective
4. **Multi-stage sequence lengths**: 128 tokens (epochs 1-2), 512 tokens (epoch 3)

---

## Hardware Requirements

### Minimum Requirements

- **GPU**: NVIDIA GPU with 40GB VRAM (e.g., A100)
- **RAM**: 32GB system memory
- **CPU**: 12+ cores recommended
- **Storage**: 100GB free disk space

### Recommended Setup

Our experiments used:
- **GPU**: 1x NVIDIA A100 (40GB)
- **RAM**: 32GB
- **CPU**: 12 cores
- **Storage**: 150GB SSD

### Training Time Estimates

| Stage | Sequence Length | Time per Epoch | Total Time |
|-------|----------------|----------------|------------|
| Epochs 1-2 | 128 tokens | ~12 hours | ~24 hours |
| Epoch 3 | 512 tokens | ~36 hours | ~36 hours |
| **Total** | - | - | **~60 hours** |

*Times are approximate and depend on hardware*

---

## Data Preparation

### Step 1: Download Datasets

We use four public Arabic corpora:

1. **OSIAN** (Open Source International Arabic News)
2. **Arabic Billion Words** (1.5B words)
3. **Arabic Wikipedia** (latest dump)
4. **OSCAR Arabic** (OSCAR 2022)

**Total**: ~17GB preprocessed, 6M+ sentences

#### Download Script

```bash
python scripts/pretraining/run_data_collection.py \
    --config configs/pretraining_config.yaml \
    --download-only
```

This will:
- Read URLs from `data/links.json`
- Download files to `data/raw/`
- Extract compressed archives
- Log progress

#### Manual Download

If automatic download fails, manually download from:
- Wikipedia: https://dumps.wikimedia.org/arwiki/
- OSIAN: Contact authors or use web scraping
- Arabic Billion Words: Available through academic channels
- OSCAR: https://oscar-project.org/

### Step 2: Preprocess Data

Preprocessing pipeline:

```bash
python scripts/pretraining/run_data_collection.py \
    --config configs/pretraining_config.yaml \
    --preprocess-only
```

#### Preprocessing Steps

1. **Diacritics Removal**
   ```python
   # Remove Arabic diacritical marks (tashkeel)
   text = remove_diacritics(text)
   ```

2. **Elongation Removal**
   ```python
   # Remove tatweel (ـ) characters
   text = text.replace('ـ', '')
   ```

3. **Punctuation Standardization**
   ```python
   # Standardize Arabic and English punctuation
   text = normalize_punctuation(text)
   ```

4. **Farasa Segmentation** *(Optional but recommended)*
   ```python
   # Apply morphological segmentation
   from farasa.segmenter import FarasaSegmenter
   segmenter = FarasaSegmenter()
   text = segmenter.segment(text)
   ```

5. **Filtering**
   - Remove empty lines
   - Remove sentences < 5 words
   - Remove sentences > 512 tokens

#### Output Structure

```
data/processed/
├── train/
│   ├── chunk_0001.txt
│   ├── chunk_0002.txt
│   └── ...
├── validation/
│   ├── chunk_0001.txt
│   └── ...
└── test/
    ├── chunk_0001.txt
    └── ...
```

---

## Tokenizer Extension

Extend ModernBERT's vocabulary with Arabic-specific tokens.

### Why Extend the Vocabulary?

- ModernBERT's tokenizer is optimized for English
- Arabic has unique morphological features
- Extension reduces subword fragmentation
- Improves representation of Arabic roots and affixes

### Step 1: Analyze Vocabulary Size

The optimal vocabulary size balances coverage and efficiency. Our analysis shows:

- Coverage plateaus around **80K tokens**
- Beyond 80K provides diminishing returns
- Consistent with other Arabic BERT models (AraBERT: 64K, MARBERT: 95K)

### Step 2: Train Extended Tokenizer

```bash
python scripts/pretraining/run_tokenizer_extension.py \
    --input-dir ./data/processed \
    --output-tokenizer ./tokenizer \
    --vocab-size 80000 \
    --min-frequency 5
```

#### Parameters

- `--vocab-size`: Number of new Arabic tokens (default: 80000)
- `--min-frequency`: Minimum token frequency (default: 5)
- `--special-tokens`: Additional special tokens if needed

#### Output

```
tokenizer/
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
└── tokenizer.json
```

### Step 3: Verify Tokenizer

Test the extended tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

# Test Arabic text
text = "محمد يذهب إلى المدرسة"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
print(f"Vocab size: {len(tokenizer)}")
```

---

## Training Process

### Configuration

Review and customize `configs/pretraining_config.yaml`:

```yaml
model:
  base_model: "answerdotai/ModernBERT-base"
  extended_vocab_size: 80000

training:
  num_epochs: 3
  batch_size: 32
  learning_rate: 1.0e-4
  sequence_length_schedule:
    - epochs: [1, 2]
      max_length: 128
    - epochs: [3]
      max_length: 512
```

### Run Training

```bash
python scripts/pretraining/run_pretraining.py \
    --config configs/pretraining_config.yaml
```

### Training Script Features

- **Mixed Precision Training** (FP16/BF16): Reduces memory usage
- **Gradient Checkpointing**: Trades computation for memory
- **Automatic Checkpointing**: Saves every 5000 steps
- **Distributed Training**: Multi-GPU support via Accelerate
- **Memory Monitoring**: Tracks RAM and VRAM usage

### Multi-Stage Training

#### Stage 1: Short Sequences (Epochs 1-2)

```yaml
sequence_length: 128
batch_size: 32
estimated_time_per_epoch: 12 hours
```

**Benefits**:
- Faster training
- Lower memory usage
- Good for learning basic patterns

#### Stage 2: Long Sequences (Epoch 3)

```yaml
sequence_length: 512
batch_size: 32  # May need to reduce if OOM
estimated_time_per_epoch: 36 hours
```

**Benefits**:
- Models longer dependencies
- Better for downstream tasks
- Matches evaluation setup

### Resume from Checkpoint

If training is interrupted:

```bash
python scripts/pretraining/run_pretraining.py \
    --config configs/pretraining_config.yaml \
    --resume-from-checkpoint ./checkpoints/checkpoint-step-25000
```

---

## Monitoring and Evaluation

### Training Metrics

Monitor during training:

1. **Loss**: Should decrease steadily
2. **Perplexity**: `exp(loss)`, lower is better
3. **Learning Rate**: Follows cosine schedule
4. **Memory Usage**: RAM and VRAM consumption

### Logging

Logs are saved to:
- Console output
- `training.log` file
- Weights & Biases (if enabled)

### Evaluation

Evaluate on validation set every 5000 steps:

```python
# Validation metrics
{
  "eval_loss": 2.45,
  "eval_perplexity": 11.59,
  "eval_runtime": 123.45,
  "eval_samples_per_second": 450.2
}
```

### Checkpoints

Checkpoints are saved in `./checkpoints/`:

```
checkpoints/
├── checkpoint-step-5000/
├── checkpoint-step-10000/
├── checkpoint-step-15000/
└── ...
```

Each checkpoint contains:
- Model weights (`pytorch_model.bin`)
- Optimizer state
- Training configuration
- RNG states (for reproducibility)

---

## Troubleshooting

### Out of Memory (OOM)

**Problem**: CUDA out of memory error

**Solutions**:

1. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 16  # Instead of 32
   ```

2. **Enable gradient checkpointing**:
   ```yaml
   training:
     gradient_checkpointing: true
   ```

3. **Use gradient accumulation**:
   ```yaml
   training:
     batch_size: 16
     gradient_accumulation_steps: 2  # Effective batch size = 32
   ```

4. **Reduce sequence length**:
   ```yaml
   training:
     sequence_length_schedule:
       - epochs: [1, 2, 3]
         max_length: 128  # Skip 512 stage
   ```

### Slow Training

**Problem**: Training is slower than expected

**Solutions**:

1. **Check data loading**:
   ```yaml
   system:
     num_workers: 8  # Increase workers
     pin_memory: true
     prefetch_factor: 4
   ```

2. **Enable mixed precision**:
   ```yaml
   training:
     mixed_precision: "fp16"  # or "bf16"
   ```

3. **Optimize dataloader**:
   - Preprocess data once, not on-the-fly
   - Use faster storage (SSD vs HDD)

### NaN Loss

**Problem**: Loss becomes NaN during training

**Solutions**:

1. **Reduce learning rate**:
   ```yaml
   training:
     learning_rate: 5.0e-5  # Instead of 1.0e-4
   ```

2. **Enable gradient clipping**:
   ```yaml
   training:
     max_grad_norm: 1.0
   ```

3. **Check data quality**:
   - Remove corrupted samples
   - Check for extreme values

### Low Disk Space

**Problem**: Running out of disk space

**Solutions**:

1. **Delete old checkpoints**:
   ```yaml
   training:
     save_total_limit: 3  # Keep only last 3
   ```

2. **Use compression**:
   ```bash
   # Compress old checkpoints
   tar -czf checkpoint-step-5000.tar.gz checkpoint-step-5000/
   rm -rf checkpoint-step-5000/
   ```

3. **Clean temporary files**:
   ```bash
   rm -rf data/interim/*
   rm -rf /tmp/*
   ```

---

## Best Practices

### 1. Start Small

Test on a small dataset first:

```bash
python scripts/pretraining/run_pretraining.py \
    --config configs/pretraining_config.yaml \
    --max-samples 10000 \
    --num-epochs 1
```

### 2. Monitor Resource Usage

Use system monitoring:

```bash
# GPU monitoring
watch -n 1 nvidia-smi

# Memory monitoring
htop
```

### 3. Regular Checkpoints

Save checkpoints frequently:

```yaml
training:
  save_steps: 2500  # Every 2500 steps
  save_total_limit: 10  # Keep last 10
```

### 4. Validation Checks

Validate regularly:

```yaml
validation:
  eval_steps: 2500
  eval_strategy: "steps"
```

### 5. Reproducibility

Set seeds for reproducibility:

```yaml
training:
  seed: 42
  data_seed: 42
```

---

## Next Steps

After pretraining:

1. **Upload to Hugging Face Hub**:
   ```python
   from transformers import AutoModel
   
   model = AutoModel.from_pretrained("./checkpoints/checkpoint-final")
   model.push_to_hub("your-org/ModernAraBERT")
   ```

2. **Fine-tune on downstream tasks**: See [BENCHMARKING.md](./BENCHMARKING.md)

3. **Share with community**: Update model card, add examples

---

## Citation

If you use this pretraining approach, please cite:

```bibtex
@inproceedings{eldamaty2026modernarabert,
  title={Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic},
  author={Eldamaty, Ahmed and Maher, Mohamed and Mostafa, Mohamed and Ashraf, Mariam and ElShawi, Radwa},
  booktitle={Proceedings of LREC-COLING 2026},
  year={2026}
}
```

---

## Support

For questions or issues:
- Open an issue on [GitHub](https://github.com/giza-data-team/ModernAraBERT/issues)
- Contact: ahmed.aldamati@gizasystems.com

