"""
Named Entity Recognition (NER) Benchmarking Module

This module provides comprehensive NER benchmarking for ModernAraBERT:
- ANERCorp dataset support with IOB2 tagging
- Custom WeightedNERTrainer with Focal Loss
- First-subtoken labeling strategy
- Micro-F1 evaluation metrics
- Memory usage tracking
- Class imbalance handling

Main components:
- ner_benchmark.py: Complete NER benchmarking framework

Key Features:
- Balanced class weights for imbalanced datasets
- Focal Loss for hard example mining
- First-subtoken-only labeling
- Word-level evaluation metrics
- Comprehensive result logging
"""

__all__ = []

