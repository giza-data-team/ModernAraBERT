"""
Sentiment Analysis Benchmarking Module

This module provides tools for benchmarking ModernAraBERT on Arabic SA tasks:
- Multiple datasets: HARD, ASTD, LABR, AJGT
- Dataset preparation and preprocessing
- Training with frozen encoders
- Comprehensive evaluation metrics

Main components:
- datasets.py: Dataset loading and preparation
- preprocessing.py: Text preprocessing utilities
- train.py: Training and evaluation functions
- sa_benchmark.py: Main benchmarking orchestrator
"""

from .datasets import (
    DATASET_CONFIGS,
    load_sentiment_dataset,
    prepare_astd_benchmark,
    prepare_labr_benchmark
)

from .preprocessing import (
    is_arabic_text,
    process_text,
    process_dataset
)

from .train import (
    train_model,
    evaluate_model
)

from .sa_benchmark import (
    run_sa_benchmark,
    get_memory_usage,
    set_seed
)

__all__ = [
    # Dataset functions
    'DATASET_CONFIGS',
    'load_sentiment_dataset',
    'prepare_astd_benchmark',
    'prepare_labr_benchmark',
    # Preprocessing functions
    'is_arabic_text',
    'process_text',
    'process_dataset',
    # Training functions
    'train_model',
    'evaluate_model',
    # Benchmarking functions
    'run_sa_benchmark',
    'get_memory_usage',
    'set_seed',
]

