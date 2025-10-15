"""
Sentiment Analysis Benchmarking Module for ModernAraBERT

This module provides the main benchmarking orchestrator for evaluating models on SA tasks:
- Supports multiple datasets: HARD, ASTD, LABR, AJGT
- Supports multiple models: ModernAraBERT, AraBERT, mBERT, etc.
- Fine-tuning with frozen encoders
- Comprehensive evaluation with Macro-F1, loss, and perplexity
- Memory usage tracking
- Result logging and JSON export
"""

import os
import logging
import random
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Optional
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)

# Import from local modules
from .datasets import (
    load_sentiment_dataset,
)
from .train import train_model, evaluate_model
from utils.memory import get_memory_usage


# Suppress tokenizers parallelism warning
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


# Model configurations
MODEL_PATHS = {
    "modernarabert": {
        "path": "gizadatateam/ModernAraBERT",
        "tokenizer_path": "gizadatateam/ModernAraBERT"
    },
    "arabert": {
        "path": "aubmindlab/bert-base-arabert",
        "tokenizer_path": "aubmindlab/bert-base-arabert"
    },
    "mbert": {
        "path": "google-bert/bert-base-multilingual-cased",
        "tokenizer_path": "google-bert/bert-base-multilingual-cased"
    },
    "arabert2": {
        "path": "aubmindlab/bert-base-arabertv2",
        "tokenizer_path": "aubmindlab/bert-base-arabertv2"
    },
    "marbert": {
        "path": "UBC-NLP/MARBERTv2",
        "tokenizer_path": "UBC-NLP/MARBERTv2"
    },
}

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_sa_benchmark(
    model_name: str,
    dataset_name: str,
    model_path: str,
    tokenizer_path: Optional[str],
    data_dir: str,
    num_labels: int,
    batch_size: int = 16,
    max_length: int = 512,
    epochs: int = 50,
    learning_rate: float = 2e-5,
    patience: int = 5,
    num_workers: int = 2,
    freeze_encoder: bool = True,
    checkpoint_path: Optional[str] = None,
    continue_from_checkpoint: bool = False,
    save_every: Optional[int] = None,
    hf_token: Optional[str] = None,
    log_dir: str = "./logs",
    results_dir: str = "./results/sa",
    seed: int = 42
) -> Dict:
    """
    Main benchmarking function for sentiment analysis.

    Args:
        model_name (str): Model identifier
        dataset_name (str): Dataset name (hard, astd, labr, ajgt)
        model_path (str): Path to pretrained model
        tokenizer_path (str): Path to tokenizer (None = use model_path)
        data_dir (str): Directory containing dataset files
        num_labels (int): Number of classes
        batch_size (int): Training batch size (default: 16)
        max_length (int): Maximum sequence length (default: 512)
        epochs (int): Number of training epochs (default: 50)
        learning_rate (float): Learning rate (default: 2e-5)
        patience (int): Early stopping patience (default: 5)
        num_workers (int): DataLoader workers (default: 2)
        freeze_encoder (bool): Freeze encoder layers (default: True)
        checkpoint_path (str): Path to save/load checkpoints
        continue_from_checkpoint (bool): Resume from checkpoint
        save_every (int): Save checkpoint every N epochs
        hf_token (str): HuggingFace token for private models
        log_dir (str): Directory for logs
        results_dir (str): Directory for saving results (default: ./results/sa)
        seed (int): Random seed (default: 42)

    Returns:
        dict: Benchmark results including metrics and configuration
    """
    # Setup
    logger = logging.getLogger(__name__)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting benchmarking for model {model_name} on dataset {dataset_name}")
    logger.info(f"Configuration: epochs={epochs}, patience={patience}")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=hf_token)
    
    # Load model configuration
    config = AutoConfig.from_pretrained(model_path, num_labels=num_labels, token=hf_token)
    model = AutoModelForSequenceClassification.from_config(config)
    
    # Replace classifier head
    hidden_size = model.config.hidden_size
    model.classifier = nn.Linear(hidden_size, num_labels)
    
    # Freeze encoder if requested
    if freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        logger.info("Encoder layers frozen, only classifier will be trained")
    
    model.to(device)
    
    # Prepare file paths
    train_file = os.path.join(data_dir, "train.txt")
    val_file = os.path.join(data_dir, "validation.txt")
    test_file = os.path.join(data_dir, "test.txt")
    
    # Determine dataset type for label mapping
    dataset_type = dataset_name.lower()
    
    # Load training data
    logger.info(f"Loading training data from {train_file}")
    train_samples = load_sentiment_dataset(
        train_file,
        tokenizer,
        None,
        num_labels=num_labels,
        max_length=max_length,
        dataset_type=dataset_type
    )
    
    if len(train_samples) == 0:
        logger.error(f"No training samples loaded for model {model_name}.")
        raise Exception("No training samples.")
    
    train_dataloader = DataLoader(
        train_samples,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Load validation data (if available)
    if dataset_name.lower() == "labr":
        logger.warning("No validation set for LABR dataset; validation will be skipped.")
        val_dataloader = None
    else:
        logger.info(f"Loading validation data from {val_file}")
        val_samples = load_sentiment_dataset(
            val_file,
            tokenizer,
            None,
            num_labels=num_labels,
            max_length=max_length,
            dataset_type=dataset_type
        )
        
        if len(val_samples) == 0:
            logger.error(f"No validation samples loaded for model {model_name}.")
            raise Exception("No validation samples.")
        
        val_dataloader = DataLoader(
            val_samples,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Log initial memory usage
    initial_memory = get_memory_usage()
    logger.info("Initial memory usage before training:")
    logger.info(f"  RAM: {initial_memory['ram_used_gb']:.2f} GB ({initial_memory['ram_percent']:.1f}%)")
    if torch.cuda.is_available():
        logger.info(f"  VRAM: {initial_memory['vram_used_gb']:.2f} GB ({initial_memory['vram_percent']:.1f}%)")
    
    # Train model
    logger.info(f"Starting training for model: {model_name}")
    model = train_model(
        model,
        train_dataloader,
        val_dataloader,
        device,
        num_epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        checkpoint_path=checkpoint_path,
        continue_from_checkpoint=continue_from_checkpoint,
        save_every=save_every
    )
    
    # Log final memory usage
    final_memory = get_memory_usage()
    logger.info("Final memory usage after training:")
    logger.info(f"  RAM: {final_memory['ram_used_gb']:.2f} GB ({final_memory['ram_percent']:.1f}%)")
    if torch.cuda.is_available():
        logger.info(f"  VRAM: {final_memory['vram_used_gb']:.2f} GB ({final_memory['vram_percent']:.1f}%)")
    
    # Load test data
    logger.info(f"Loading test data from {test_file}")
    test_samples = load_sentiment_dataset(
        test_file,
        tokenizer,
        None,
        num_labels=num_labels,
        max_length=max_length,
        dataset_type=dataset_type
    )
    
    test_dataloader = DataLoader(
        test_samples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    macro_f1, report, preds, true_labels, confidences, avg_eval_loss, perplexity = evaluate_model(
        model, test_dataloader, device
    )
    
    logger.info(f"{model_name} Evaluation Macro-F1: {macro_f1:.4f}")
    logger.info(f"{model_name} Classification Report:\n{report}")
    logger.info(f"{model_name} Average Eval Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.4f}")
    
    logger.info(f"\n{model_name} Evaluation Results:")
    logger.info(f"  Macro-F1: {macro_f1:.4f}")
    logger.info(f"  Average Loss: {avg_eval_loss:.4f}")
    logger.info(f"  Perplexity: {perplexity:.4f}")
    logger.info("\nClassification Report:")
    logger.info(report)
    
    # Log sample predictions
    logger.info("\nSample predictions (first 10):")
    logger.info(f"{'Index':<6}{'Ground Truth':<15}{'Prediction':<15}{'Confidence':<12}")
    for i in range(min(10, len(true_labels))):
        logger.info(f"{i:<6}{true_labels[i]:<15}{preds[i]:<15}{confidences[i]:<12.4f}")
    
    # Save final checkpoint if requested
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
        tokenizer.save_pretrained(os.path.dirname(checkpoint_path))
        logger.info(f"Final model and tokenizer saved to {checkpoint_path}")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Prepare results dictionary
    results = {
        "configuration": {
            "model": model_name,
            "dataset": dataset_name,
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_sequence_length": max_length,
            "freeze_layers": freeze_encoder,
            "seed": seed
        },
        "hardware_usage": {
            "initial_memory": {
                "ram_used_gb": float(initial_memory['ram_used_gb']),
                "ram_percent": float(initial_memory['ram_percent']),
                "vram_used_gb": float(initial_memory['vram_used_gb']),
                "vram_total_gb": float(initial_memory['vram_total_gb']),
                "vram_percent": float(initial_memory['vram_percent'])
            },
            "final_memory": {
                "ram_used_gb": float(final_memory['ram_used_gb']),
                "ram_percent": float(final_memory['ram_percent']),
                "vram_used_gb": float(final_memory['vram_used_gb']),
                "vram_total_gb": float(final_memory['vram_total_gb']),
                "vram_percent": float(final_memory['vram_percent'])
            }
        },
        "results": {
            model_name: {
                "macro_f1": float(macro_f1),
                "avg_eval_loss": float(avg_eval_loss),
                "perplexity": float(perplexity)
            }
        }
    }
    
    # Save results to JSON
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"sa_benchmark_{model_name}_{dataset_name}_{epochs}ep_p{patience}_{timestamp}_results.json"
    result_filepath = os.path.join(results_dir, result_filename)
    with open(result_filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {result_filepath}")
    logger.info(f"Detailed results saved to {result_filepath}")
    
    # Log summary
    logger.info("\nBenchmarking Complete!")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Macro-F1: {macro_f1:.4f}")
    logger.info(f"  Perplexity: {perplexity:.4f}")
    logger.info("\nHardware Usage:")
    logger.info(f"  Initial RAM: {initial_memory['ram_used_gb']:.2f} GB ({initial_memory['ram_percent']:.1f}%)")
    logger.info(f"  Final RAM: {final_memory['ram_used_gb']:.2f} GB ({final_memory['ram_percent']:.1f}%)")
    if torch.cuda.is_available():
        logger.info(f"  Initial VRAM: {initial_memory['vram_used_gb']:.2f} GB ({initial_memory['vram_percent']:.1f}%)")
        logger.info(f"  Final VRAM: {final_memory['vram_used_gb']:.2f} GB ({final_memory['vram_percent']:.1f}%)")
    
    return results
