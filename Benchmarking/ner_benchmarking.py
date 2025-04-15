"""
Named Entity Recognition (NER) Benchmarking Framework for Arabic Language Models

This script provides a comprehensive framework for benchmarking transformer-based models
on Named Entity Recognition (NER) tasks for Arabic language. It supports:

1. Fine-tuning pre-trained models on NER datasets with configurable parameters
2. Benchmarking different models (ModernBERT, AraBERT) on standard NER datasets
3. Detailed evaluation metrics (precision, recall, F1, accuracy)
4. Configurable fine-tuning strategies (full model or classification head only)
5. Structured results saving for paper publication and analysis
6. Memory usage tracking (RAM and VRAM) for model comparison

The implementation follows best practices for NER evaluation and provides detailed logging
of training progress and results suitable for academic publication.

Authors: Mohamed Maher, Ahmed Samy, Mariam Ashraf, Mohamed Mostafa, Ali Nasser, Ali Sameh
Date: April 2025
License: [Appropriate License]
"""

# Set tokenizers parallelism environment variable to avoid deadlocks
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import sys
import time
import psutil
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# Set up constants for model paths
MODEL_PATHS = {
    "modernbert": {
        "model": "./model_checkpoints/checkpoint_step_13000/",
        "tokenizer": "./Tokenizer",
    },
    "arabert": {
        "model": "aubmindlab/bert-base-arabert",
        "tokenizer": "aubmindlab/bert-base-arabert",
    },
}

# Default paths for data and outputs
DEFAULT_DATA_DIR = "./Data/"
DEFAULT_OUTPUT_DIR = "./ner_results"
DEFAULT_LOG_DIR = "./logs"
TOKEN_IGNORED_LABEL = -100  

def setup_logging(log_filepath: str) -> None:
    """
    Set up logging configuration with file and console output.
    
    Args:
        log_filepath: Path to save the log file
    """
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filepath),
        ],
        force=True,
    )
    logging.info(f"Logging initialized. Saving logs to: {log_filepath}")


def get_unique_output_path(base_dir: str, model_name: str, dataset_name: str, epochs: int) -> str:
    """
    Generate a unique output path based on experiment parameters and timestamp.
    
    Args:
        base_dir: Base directory for outputs
        model_name: Name of the model being benchmarked
        dataset_name: Name of the dataset used
        epochs: Number of training epochs
        
    Returns:
        Unique path string for this experiment run
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_safe = dataset_name.replace("/", "_")
    directory_name = f"{model_name}_{dataset_name_safe}_{epochs}ep_{timestamp}"
    return os.path.join(base_dir, directory_name)


def process_ner_dataset(
    tokenizer,
    dataset_name: str,
) -> Tuple[Dict[str, Dataset], Dict[str, int], Dict[int, str]]:
    """
    Process a NER dataset for model training and evaluation.
    
    This function:
    1. Loads a Named Entity Recognition dataset
    2. Processes it for transformer model consumption
    3. Creates train/val/test splits if needed
    4. Handles tokenization and label mapping
    
    Args:
        tokenizer: Tokenizer instance for the model
        dataset_name: Name/path of the dataset to load
        
    Returns:
        Tuple containing:
            - processed_dataset: Dictionary with train/val/test splits
            - label2id: Mapping from NER tags to numeric IDs
            - id2label: Mapping from numeric IDs to NER tags
    """
    logging.info(f"Processing dataset: {dataset_name}")
    
    try:
        from datasets import load_dataset, Dataset, DatasetDict
        
        # Load the dataset
        raw_dataset = load_dataset(dataset_name)
        logging.info(f"Dataset loaded with keys: {raw_dataset.keys()}")
        
        # Convert to the expected sentence format
        dataset_sentences = transform_to_sentence_format(raw_dataset)
        dataset = DatasetDict({
            "train": Dataset.from_list(dataset_sentences["train"]),
            "test": Dataset.from_list(dataset_sentences["test"]),
        })
        
        # Create a validation split if needed
        if "validation" not in dataset:
            train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
            dataset["train"] = train_val_split["train"]
            dataset["validation"] = train_val_split["test"]
        
        # Create label mappings
        label2id, id2label = get_label_mapping(dataset["train"])
        logging.info(f"Created label mappings with {len(label2id)} unique labels")
        
        # Process dataset for transformer model
        tokenized_datasets = dataset.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id),
            batched=True,
        )
        
        # Log dataset statistics
        for split in tokenized_datasets:
            logging.info(f"{split} split: {len(tokenized_datasets[split])} examples")
            
        return tokenized_datasets, label2id, id2label
        
    except Exception as e:
        logging.error(f"Error processing dataset: {e}")
        raise


def transform_to_sentence_format(dataset):
    """
    Convert token-level dataset to sentence-level format.
    
    Args:
        dataset: Dataset with token-level annotations
        
    Returns:
        Dictionary of sentence-level data for each split
    """
    dataset_sentences = {}
    
    for split_name in dataset.keys():
        sentences = []
        sentence = {"tokens": [], "ner_tags": []}
        
        for example in dataset[split_name]:
            token, ner_tag = example["word"], example["tag"]
            
            if token == ".":
                if sentence["tokens"]:
                    sentences.append(sentence)
                sentence = {"tokens": [], "ner_tags": []}
            else:
                sentence["tokens"].append(token)
                sentence["ner_tags"].append(ner_tag)
                
        if sentence["tokens"]:
            sentences.append(sentence)
            
        dataset_sentences[split_name] = sentences
        
    return dataset_sentences


def get_label_mapping(sentences):
    """
    Create label-to-id and id-to-label mappings from tagged sentences.
    
    Args:
        sentences: List of examples with "ner_tags" field
        
    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    unique_labels = set()
    
    for example in sentences:
        unique_labels.update(example["ner_tags"])
        
    unique_labels = sorted(unique_labels)
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return label2id, id2label


def tokenize_and_align_labels(examples, tokenizer, label2id):
    """
    Tokenize inputs and align labels properly for NER tasks.
    
    Args:
        examples: Examples to process
        tokenizer: Tokenizer to use
        label2id: Mapping from string labels to IDs
        
    Returns:
        Processed examples with aligned labels
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    labels = []
    
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens
            if word_idx is None:
                label_ids.append(TOKEN_IGNORED_LABEL)
            # First token of a word
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            # Continuation tokens
            else:
                current_label = label[word_idx]
                # Convert B- tags to I- tags for continuation tokens
                if current_label.startswith("B-"):
                    entity_type = current_label[2:]
                    i_tag = f"I-{entity_type}"
                    
                    if i_tag in label2id:
                        label_ids.append(label2id[i_tag])
                    else:
                        label_ids.append(TOKEN_IGNORED_LABEL)
                else:
                    label_ids.append(label2id[current_label])
                    
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_ner_model(
    model, 
    train_dataset, 
    eval_dataset, 
    test_dataset=None, 
    output_dir=DEFAULT_OUTPUT_DIR, 
    num_epochs=3, 
    learning_rate=2e-5,
    batch_size=8,
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    save_steps=1000,
    eval_steps=100
):
    """
    Train a NER model using the Hugging Face Transformers Trainer API.
    
    Args:
        model: Pre-trained transformer model for token classification
        train_dataset: Dataset containing training examples
        eval_dataset: Dataset containing validation examples
        test_dataset: Dataset containing test examples (optional)
        output_dir: Directory to save the trained model
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        batch_size: Batch size for training and evaluation
        gradient_accumulation_steps: Number of steps to accumulate gradients
        weight_decay: L2 regularization weight
        warmup_ratio: Ratio of steps for learning rate warmup
        logging_steps: Number of steps between logging updates
        save_steps: Number of steps between model checkpoints
        eval_steps: Number of steps between evaluations
        
    Returns:
        Tuple containing:
            - model: The trained NER model
            - eval_results: Validation metrics
            - test_metrics: Test metrics if test_dataset provided
    """
    initial_memory = get_memory_usage()
    logging.info("Initial memory usage before training:")
    logging.info(f"  RAM: {initial_memory['ram_used_gb']:.2f} GB ({initial_memory['ram_percent']:.1f}%)")
    if torch.cuda.is_available():
        logging.info(f"  VRAM: {initial_memory['vram_used_gb']:.2f} GB ({initial_memory['vram_percent']:.1f}%)")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,      
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_total_limit=2,  
        report_to=[],  
        fp16=torch.cuda.is_available(),  
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=4, 
    )

    # Log training configuration
    logging.info("Training configuration:")
    logging.info(f"  Epochs: {num_epochs}")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Weight decay: {weight_decay}")
    logging.info(f"  FP16: {torch.cuda.is_available()}")
    logging.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Initialize trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_ner_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model and catch any exceptions
    peak_memory = initial_memory.copy()
    try:
        logging.info("Starting model training...")
        train_result = trainer.train()
        
        # Get peak memory usage during training
        current_memory = get_memory_usage()
        for key in current_memory:
            if key.endswith('_gb') or key.endswith('_percent'):
                peak_memory[key] = max(peak_memory.get(key, 0), current_memory[key])
        
        # Log training metrics
        logging.info(f"Training completed in {train_result.metrics.get('train_runtime', 0):.2f} seconds")
        logging.info(f"Training loss: {train_result.metrics.get('train_loss', 0):.4f}")
        logging.info(f"Training steps per second: {train_result.metrics.get('train_steps_per_second', 0):.2f}")
        
        trainer.save_model(output_dir)
        logging.info(f"Model saved to {output_dir}")
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise

    logging.info("Peak memory usage during training:")
    logging.info(f"  RAM: {peak_memory['ram_used_gb']:.2f} GB ({peak_memory['ram_percent']:.1f}%)")
    if torch.cuda.is_available():
        logging.info(f"  VRAM: {peak_memory['vram_used_gb']:.2f} GB ({peak_memory['vram_percent']:.1f}%)")

    # Evaluate on validation set
    logging.info("Evaluating model on validation set...")
    eval_results = trainer.evaluate()
    
    logging.info("Validation metrics:")
    for key, value in eval_results.items():
        logging.info(f"  {key}: {value:.4f}")

    # Test the model if test_dataset is provided
    test_metrics = None
    if test_dataset is not None:
        logging.info("Testing model on test set...")
        try:
            test_results = trainer.predict(test_dataset)
            test_metrics = test_results.metrics
            
            # Add memory usage metrics to test metrics
            test_metrics["peak_ram_gb"] = peak_memory["ram_used_gb"]
            test_metrics["peak_ram_percent"] = peak_memory["ram_percent"]
            if torch.cuda.is_available():
                test_metrics["peak_vram_gb"] = peak_memory["vram_used_gb"]
                test_metrics["peak_vram_percent"] = peak_memory["vram_percent"]
                test_metrics["vram_total_gb"] = peak_memory["vram_total_gb"]
            
            logging.info("Test metrics:")
            for key, value in test_metrics.items():
                if isinstance(value, float):
                    logging.info(f"  {key}: {value:.4f}")
                else:
                    logging.info(f"  {key}: {value}")
        except Exception as e:
            logging.error(f"Testing failed with error: {str(e)}")
    
    return model, eval_results, test_metrics


def compute_ner_metrics(pred):
    """
    Compute evaluation metrics for NER task from model predictions.
    
    Args:
        pred: Prediction object containing predictions and labels
            
    Returns:
        Dictionary containing precision, recall, F1, and accuracy metrics
    """
    # Extract predictions and labels
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens) from predictions and labels
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != TOKEN_IGNORED_LABEL]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for l in label if l != TOKEN_IGNORED_LABEL]
        for label in labels
    ]

    # Flatten the predictions and labels for metric calculation
    flat_predictions = [p for pred in true_predictions for p in pred]
    flat_labels = [l for lab in true_labels for l in lab]

    # Handle edge case where no valid predictions are found
    if not flat_predictions or not flat_labels:
        logging.warning("No valid predictions or labels found for metric calculation!")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    # Calculate metrics using sklearn's functions with weighted average
    # to handle class imbalance common in NER tasks
    accuracy = accuracy_score(flat_labels, flat_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_labels,
        flat_predictions,
        average='weighted', 
        zero_division=0     
    )

    # Add small epsilon to avoid exact 1.0 values which might indicate issues
    epsilon = 1e-10
    metrics = {
        "accuracy": min(1.0 - epsilon, float(accuracy)),
        "precision": min(1.0 - epsilon, float(precision)),
        "recall": min(1.0 - epsilon, float(recall)),
        "f1": min(1.0 - epsilon, float(f1))
    }
    
    return metrics


def run_inference_test(model, tokenizer, test_sentence="تقع مدينة باريس في فرنسا"):
    """
    Run a sample inference test on a given sentence.
    
    Args:
        model: The fine-tuned NER model
        tokenizer: Tokenizer for the model
        test_sentence: Arabic test sentence to analyze
        
    Returns:
        List of (token, label) tuples for the recognized entities
    """
    device = model.device
    
    # Tokenize the test sentence
    inputs = tokenizer(
        test_sentence,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run model inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Process predictions
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    predicted_labels = [model.config.id2label.get(p, "O") for p in predictions]
    
    # Extract meaningful token-label pairs (excluding special tokens)
    result = []
    for token, label in zip(tokens, predicted_labels):
        if token not in tokenizer.all_special_tokens:
            result.append((token, label))
            
    # Log results
    logging.info(f"Test sentence: {test_sentence}")
    logging.info("Predicted entities:")
    for token, label in result:
        logging.info(f"  {token}: {label}")
        
    return result


def save_benchmark_results(
    result_filepath, 
    model_name, 
    dataset_name, 
    epochs, 
    learning_rate,
    batch_size, 
    fine_tune,
    metrics
):
    """
    Save benchmark results to a JSON file.
    
    Args:
        result_filepath: Path to save results
        model_name: Name of the benchmarked model
        dataset_name: Name of the dataset used
        epochs: Number of training epochs
        learning_rate: Learning rate used
        batch_size: Batch size used
        fine_tune: Fine-tuning strategy
        metrics: Dictionary containing evaluation metrics
    """
    # Extract metrics
    accuracy = metrics.get("test_accuracy", 0.0)
    f1 = metrics.get("test_f1", 0.0)
    precision = metrics.get("test_precision", 0.0)
    recall = metrics.get("test_recall", 0.0)
    loss = metrics.get("test_loss", 0.0)
    runtime = metrics.get("test_runtime", 0.0)
    samples_per_second = metrics.get("test_samples_per_second", 0.0)
    steps_per_second = metrics.get("test_steps_per_second", 0.0)
    
    # Extract memory metrics
    peak_ram_gb = metrics.get("peak_ram_gb", 0.0)
    peak_ram_percent = metrics.get("peak_ram_percent", 0.0)
    peak_vram_gb = metrics.get("peak_vram_gb", 0.0)
    peak_vram_percent = metrics.get("peak_vram_percent", 0.0)
    vram_total_gb = metrics.get("vram_total_gb", 0.0)
    
    # Create results structure
    results_data = {
        "configuration": {
            "model": model_name,
            "dataset": dataset_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "fine_tune": fine_tune,
            "timestamp": datetime.now().isoformat(),
        },
        "results": {
            "accuracy": max(0.0, min(1.0, float(accuracy))),
            "f1_score": max(0.0, min(1.0, float(f1))),
            "precision": max(0.0, min(1.0, float(precision))),
            "recall": max(0.0, min(1.0, float(recall))),
            "loss": float(loss) if loss >= 0 else 0.0,
            "runtime_seconds": float(runtime),
            "samples_per_second": float(samples_per_second),
            "steps_per_second": float(steps_per_second),
        },
        "memory_usage": {
            "peak_ram_gb": float(peak_ram_gb),
            "peak_ram_percent": float(peak_ram_percent),
            "peak_vram_gb": float(peak_vram_gb),
            "peak_vram_percent": float(peak_vram_percent),
            "vram_total_gb": float(vram_total_gb),
        }
    }

    # Ensure metrics are valid
    for metric_name, value in results_data["results"].items():
        if not isinstance(value, (int, float)) or (
            metric_name != "loss" and metric_name != "runtime_seconds" 
            and not 0 <= value <= 1 and "per_second" not in metric_name
        ):
            logging.warning(f"Invalid {metric_name} value: {value}. Setting to 0.0")
            results_data["results"][metric_name] = 0.0

    # Save to file
    os.makedirs(os.path.dirname(result_filepath), exist_ok=True)
    with open(result_filepath, "w") as f:
        json.dump(results_data, f, indent=2)
        
    logging.info(f"Benchmark results saved to {result_filepath}")
    return results_data


def get_memory_usage():
    """
    Get current RAM and VRAM usage statistics.
    
    Returns:
        Dict containing memory usage information:
            - ram_used_gb: RAM currently in use (GB)
            - ram_percent: Percentage of total RAM in use
            - vram_used_gb: VRAM currently in use (GB)
            - vram_total_gb: Total VRAM available (GB)
            - vram_percent: Percentage of VRAM in use
    """
    memory_stats = {}
    
    # Get RAM usage
    process = psutil.Process(os.getpid())
    ram_used_bytes = process.memory_info().rss  # Resident Set Size
    ram_used_gb = ram_used_bytes / (1024 ** 3)
    memory_stats["ram_used_gb"] = ram_used_gb
    memory_stats["ram_percent"] = psutil.virtual_memory().percent
    
    # Get VRAM usage if CUDA is available
    if torch.cuda.is_available():
        try:
            vram_used_bytes = torch.cuda.memory_allocated()
            vram_total_bytes = torch.cuda.get_device_properties(0).total_memory
            
            vram_used_gb = vram_used_bytes / (1024 ** 3)
            vram_total_gb = vram_total_bytes / (1024 ** 3)
            vram_percent = (vram_used_bytes / vram_total_bytes) * 100
            
            memory_stats["vram_used_gb"] = vram_used_gb
            memory_stats["vram_total_gb"] = vram_total_gb
            memory_stats["vram_percent"] = vram_percent
        except Exception as e:
            logging.warning(f"Could not get VRAM usage: {e}")
            memory_stats["vram_used_gb"] = 0.0
            memory_stats["vram_total_gb"] = 0.0
            memory_stats["vram_percent"] = 0.0
    else:
        memory_stats["vram_used_gb"] = 0.0
        memory_stats["vram_total_gb"] = 0.0
        memory_stats["vram_percent"] = 0.0
    
    return memory_stats


def parse_arguments():
    """
    Parse command line arguments for NER benchmarking.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Benchmark NER models for Arabic language",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="modernbert",
        choices=["modernbert", "arabert"],
        help="Model architecture to benchmark"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="asas-ai/ANERCorp",
        help="HuggingFace dataset name for NER benchmark"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=2e-5,
        help="Learning rate for fine-tuning"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--gradient-accumulation", 
        type=int, 
        default=1,
        help="Number of gradient accumulation steps"
    )
    
    # Fine-tuning strategy
    parser.add_argument(
        "--fine-tune",
        type=str,
        choices=["full", "head-only"],
        default="head-only",
        help="Fine-tune the full model or only the classification head"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",







def main():
    """Main function to run NER benchmarking"""
    args = parse_arguments()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Generate unique run IDs and paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.model}_{args.dataset.replace('/', '_')}_{timestamp}"
    
    log_filepath = os.path.join(args.log_dir, f"{run_id}.log")
    model_output_dir = os.path.join(args.output_dir, run_id)
    result_filepath = os.path.join(args.output_dir, f"{run_id}_results.json")
    
    # Set up logging
    setup_logging(log_filepath)
    
    # Log the configuration
    logging.info("=" * 80)
    logging.info(f"Starting NER Benchmark: {run_id}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Fine-tuning strategy: {args.fine_tune}")
    logging.info("Training configuration:")
    logging.info(f"  Epochs: {args.epochs}")
    logging.info(f"  Learning rate: {args.learning_rate}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Gradient accumulation steps: {args.gradient_accumulation}")
    logging.info("=" * 80)
    
    # Start benchmark timer
    start_time = time.time()
    
    try:
        # Determine if using local files
        #! to be updated after uploading the model into huggingface hub
        is_local = args.model == "modernbert"
        
        # Get model and tokenizer paths
        model_path = MODEL_PATHS[args.model]["model"]
        tokenizer_path = MODEL_PATHS[args.model]["tokenizer"]
        
        # Load tokenizer
        logging.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            local_files_only=is_local
        )
        
        # Process dataset
        logging.info(f"Processing dataset: {args.dataset}")
        dataset, label2id, id2label = process_ner_dataset(
            tokenizer, 
            args.dataset
        )
        
        # Load pre-trained model
        logging.info(f"Loading model from {model_path}")
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
        
        # Apply fine-tuning strategy
        if args.fine_tune == "head-only":
            logging.info("Freezing base model parameters for head-only fine-tuning")
            for param in model.base_model.parameters():
                param.requires_grad = False
                
        # Move model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        model.to(device)
        
        # Sample parameters before training for verification
        if args.fine_tune == "full":
            base_param = next(model.base_model.parameters()).clone().detach()[0, 0].item()
            logging.info(f"Sample base parameter before training: {base_param}")
        
        # Train and evaluate model
        logging.info(f"Starting fine-tuning for {args.epochs} epochs")
        model, _, test_metrics = train_ner_model(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            test_dataset=dataset["test"],
            output_dir=model_output_dir,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation
        )
        
        # Verify parameter updates
        if args.fine_tune == "full":
            new_base_param = next(model.base_model.parameters()).clone().detach()[0, 0].item()
            logging.info(f"Sample base parameter after training: {new_base_param}")
            logging.info(f"Base parameters changed: {base_param != new_base_param}")
        
        # Save benchmark results
        results = save_benchmark_results(
            result_filepath,
            args.model,
            args.dataset,
            args.epochs,
            args.learning_rate,
            args.batch_size,
            args.fine_tune,
            test_metrics
        )
        
        # Run inference test if requested
        if args.inference_test:
            logging.info("Running sample inference test")
            run_inference_test(model, tokenizer)
        
        # Log benchmark completion
        total_time = time.time() - start_time
        logging.info(f"Benchmark completed in {total_time:.2f} seconds")
        logging.info(f"Final F1 Score: {results['results']['f1_score']:.4f}")
        
    except Exception as e:
        logging.error(f"Benchmark failed with error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())