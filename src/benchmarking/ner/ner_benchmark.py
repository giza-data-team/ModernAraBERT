"""
Named Entity Recognition (NER) Benchmarking Framework for Arabic Language Models

This script provides a comprehensive framework for benchmarking transformer-based models
on Named Entity Recognition (NER) tasks for Arabic language. It supports:

1. Fine-tuning pre-trained models on NER datasets with configurable parameters
2. Benchmarking different models (ModernBERT, AraBERT) on standard NER datasets
3. Detailed evaluation metrics (precision, recall, macro-F1, accuracy)
4. Configurable fine-tuning strategies (full model or classification head only)
5. Structured results saving for paper publication and analysis
6. Memory usage tracking (RAM and VRAM) for model comparison

The implementation follows best practices for NER evaluation and provides detailed logging
of training progress and results suitable for academic publication.

IMPORTANT: This script uses macro-F1 score as the primary evaluation metric instead of 
weighted F1 or micro F1. Macro-F1 treats all entity classes equally by computing the 
unweighted mean of per-class F1 scores, which provides a more balanced evaluation for 
imbalanced NER datasets where some entity types are much rarer than others.

EVALUATION APPROACH: The script uses first-subtoken-only labeling during training and 
word-level evaluation. Labels are assigned only to the first subtoken of each word, 
with continuation subtokens marked as ignored. Evaluation metrics are computed at the 
word level using predictions from first subtokens only, ensuring proper NER evaluation.

Authors: Mohamed Maher, Abo Samy, Mariam Ashraf, Mohamed Mostafa, Ali Nasser, Ali Sameh
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
from datetime import datetime
from src.utils.memory import get_memory_usage
from typing import Dict, Tuple
from src.utils.logging import setup_logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
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
    "modernarabert": {
        "model": "gizadatateam/ModernAraBERT",
        "tokenizer": "gizadatateam/ModernAraBERT",
    },
    "arabert": {
        "model": "aubmindlab/bert-base-arabert",
        "tokenizer": "aubmindlab/bert-base-arabert",
    },
    "mbert": {
        "model": "bert-base-multilingual-cased",
        "tokenizer": "bert-base-multilingual-cased",
    },
    "arabert2": {
        "model": "aubmindlab/bert-base-arabertv02",
        "tokenizer": "aubmindlab/bert-base-arabertv02",
    },
    "marbert": {
        "model": "UBC-NLP/MARBERTv2",
        "tokenizer": "UBC-NLP/MARBERTv2",
    },
    "camel": {
        "model": "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
        "tokenizer": "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
    },
}
DEFAULT_OUTPUT_DIR = "./data/benchmarking/ner"
TOKEN_IGNORED_LABEL = -100  # Label for ignored tokens during training

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


def compute_ner_class_weights(dataset, label2id):
    """
    Compute balanced class weights for imbalanced NER datasets.
    
    This function addresses the severe class imbalance in NER datasets where
    "O" (Outside) tags typically dominate (80%+) while entity tags are rare.
    By computing inverse frequency weights, we give more importance to rare
    entity classes during training.
    
    Args:
        dataset: Training dataset with tokenized examples
        label2id: Mapping from string labels to numeric IDs
        
    Returns:
        torch.FloatTensor: Class weights for each label ID, ordered by label ID
    """
    logging.info("Computing class weights for imbalanced NER dataset...")
    
    # Extract all labels from the training dataset
    all_labels = []
    for example in dataset:
        labels = example.get('labels', [])
        # Filter out ignored tokens (-100) used for special tokens
        valid_labels = [label for label in labels if label != TOKEN_IGNORED_LABEL]
        all_labels.extend(valid_labels)
    
    if not all_labels:
        logging.warning("No valid labels found for class weight computation!")
        return torch.ones(len(label2id))
    
    # Get unique class labels (must be numpy array for sklearn)
    unique_classes = np.array(sorted(list(label2id.values())))
    
    # Compute balanced class weights using sklearn
    try:
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=all_labels
        )
        
        # Make weights more aggressive for extremely imbalanced NER datasets
        # Square root instead of power 1.5 to reduce extremeness
        class_weights = class_weights ** 0.5
        
        # Ensure minimum weight for entity classes (non-O tags) but more balanced
        id2label_reverse = {v: k for k, v in label2id.items()}
        for i, class_id in enumerate(unique_classes):
            label_name = id2label_reverse.get(class_id, f"ID_{class_id}")
            if label_name != "O":  # If it's an entity class
                class_weights[i] = max(class_weights[i], 1.5)  # Minimum weight of 1.5 for entities
        
        # Convert to torch tensor
        weight_tensor = torch.FloatTensor(class_weights)
        
        # Log class distribution and weights for analysis
        from collections import Counter
        label_counts = Counter(all_labels)
        
        logging.info("Class distribution and computed weights (aggressive weighting for entities):")
        for i, (class_id, weight) in enumerate(zip(unique_classes, class_weights)):
            label_name = id2label_reverse.get(class_id, f"ID_{class_id}")
            count = label_counts.get(class_id, 0)
            percentage = (count / len(all_labels)) * 100
            logging.info(f"  {label_name}: {count:,} samples ({percentage:.1f}%) -> weight: {weight:.4f}")
        
        logging.info(f"Aggressive class weights computed. Weight range: {weight_tensor.min():.4f} to {weight_tensor.max():.4f}")
        return weight_tensor
        
    except Exception as e:
        logging.error(f"Failed to compute class weights: {e}")
        logging.info("Falling back to uniform weights")
        return torch.ones(len(label2id))


class WeightedNERTrainer(Trainer):
    """
    Custom Trainer with Focal Loss for extremely imbalanced NER datasets.
    
    This trainer implements Focal Loss which dynamically focuses on hard examples
    and reduces the impact of easy examples (like abundant "O" tags), making it
    particularly effective for highly imbalanced NER datasets.
    """
    
    def __init__(self, class_weights=None, focal_alpha=0.25, focal_gamma=3.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if class_weights is not None:
            logging.info(f"Initialized WeightedNERTrainer with Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
            logging.info(f"Class weights shape: {class_weights.shape}, range: {class_weights.min():.4f} to {class_weights.max():.4f}")
        else:
            logging.info(f"Initialized WeightedNERTrainer with Focal Loss (alpha={focal_alpha}, gamma={focal_gamma}) - no class weights")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute Focal Loss for token classification.
        
        Focal Loss = -α(1-pt)^γ * log(pt)
        where pt is the probability of the correct class
        
        Args:
            model: The model being trained
            inputs: Input batch containing input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in the batch (for newer transformers versions)
            
        Returns:
            Loss tensor, optionally with model outputs
        """
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute Focal Loss if labels are provided
        if labels is not None:
            # Move tensors to the same device
            device = logits.device
            
            # Flatten logits and labels for loss computation
            # logits: (batch_size, seq_len, num_labels) -> (batch_size * seq_len, num_labels)
            # labels: (batch_size, seq_len) -> (batch_size * seq_len,)
            flat_logits = logits.view(-1, model.config.num_labels)
            flat_labels = labels.view(-1)
            
            # Create mask for non-ignored tokens
            mask = (flat_labels != TOKEN_IGNORED_LABEL)
            
            if mask.sum() == 0:
                # No valid tokens to compute loss on
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # Filter out ignored tokens
                valid_logits = flat_logits[mask]
                valid_labels = flat_labels[mask]
                
                # Compute probabilities
                log_probs = torch.nn.functional.log_softmax(valid_logits, dim=-1)
                probs = torch.exp(log_probs)
                
                # Get probabilities for the correct class
                correct_class_probs = probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
                
                # Compute focal weight: (1 - pt)^gamma
                focal_weight = (1 - correct_class_probs) ** self.focal_gamma
                
                # Apply class weights if provided
                if self.class_weights is not None:
                    weights = self.class_weights.to(device)
                    class_weights_for_batch = weights[valid_labels]
                    focal_weight = focal_weight * class_weights_for_batch
                
                # Apply alpha weighting
                alpha_weight = self.focal_alpha
                
                # Compute focal loss: -α * (1-pt)^γ * log(pt)
                focal_loss = -alpha_weight * focal_weight * log_probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
                
                # Average the loss
                loss = focal_loss.mean()
        else:
            loss = outputs.get("loss")
        
        return (loss, outputs) if return_outputs else loss


def tokenize_and_align_labels(examples, tokenizer, label2id):
    """
    Tokenize inputs and align labels for NER tasks using first-subtoken-only labeling.
    
    This approach assigns labels only to the first subtoken of each word and ignores
    all continuation subtokens. This simplifies the alignment and makes evaluation
    cleaner by working at the word level.
    
    Args:
        examples: Examples to process
        tokenizer: Tokenizer to use
        label2id: Mapping from string labels to IDs
        
    Returns:
        Processed examples with aligned labels (first subtoken only)
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
            # Special tokens (CLS, SEP, PAD, etc.)
            if word_idx is None:
                label_ids.append(TOKEN_IGNORED_LABEL)
            # First subtoken of a word - assign the actual label
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            # Continuation subtokens of the same word - ignore
            else:
                label_ids.append(TOKEN_IGNORED_LABEL)
                    
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_ner_model(
    model, 
    train_dataset, 
    eval_dataset, 
    test_dataset=None, 
    id2label=None,
    output_dir=DEFAULT_OUTPUT_DIR, 
    num_epochs=3, 
    learning_rate=2e-5,
    batch_size=8,
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    save_steps=1000,
    eval_steps=100,
    patience=8
):
    """
    Train a NER model using the Hugging Face Transformers Trainer API.
    
    Args:
        model: Pre-trained transformer model for token classification
        train_dataset: Dataset containing training examples
        eval_dataset: Dataset containing validation examples
        test_dataset: Dataset containing test examples (optional)
        id2label: Mapping from numeric IDs to string labels for detailed metrics
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
    # Get initial memory usage
    initial_memory = get_memory_usage()
    logging.info("Initial memory usage before training:")
    logging.info(f"  RAM: {initial_memory['ram_used_gb']:.2f} GB ({initial_memory['ram_percent']:.1f}%)")
    if torch.cuda.is_available():
        logging.info(f"  VRAM: {initial_memory['vram_used_gb']:.2f} GB ({initial_memory['vram_percent']:.1f}%)")
    
    # Configure training arguments with advanced optimizations
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        # eval_steps=eval_steps,
        save_strategy="epoch",
        # save_steps=save_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",  # Use macro-F1 to focus on entity class performance
        greater_is_better=True,      # Higher macro-F1 is better      
        weight_decay=0.01,          # Increased weight decay for better generalization
        warmup_ratio=0.1,           # Increased warmup for stable training
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch",
        # logging_steps=logging_steps,
        save_total_limit=2,  # Keep only the 2 best checkpoints
        report_to=[],  # Disable reporting to external services
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=4,  # Parallel data loading
        optim="adamw_torch",       # Use AdamW optimizer
        lr_scheduler_type="cosine", # Cosine learning rate schedule
        max_grad_norm=1.0,         # Gradient clipping for stability
    )

    # Log training configuration
    logging.info("Training configuration:")
    logging.info(f"  Epochs: {num_epochs}")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Weight decay: {weight_decay}")
    logging.info(f"  FP16: {torch.cuda.is_available()}")
    logging.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Create metrics computation function with label mappings
    compute_metrics_fn = create_compute_ner_metrics(id2label) if id2label else None
    
    # Compute class weights for imbalanced dataset
    logging.info("Computing class weights for imbalanced dataset...")
    class_weights = compute_ner_class_weights(train_dataset, model.config.label2id)
    
    # Initialize weighted trainer with early stopping
    trainer = WeightedNERTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],  # Use configurable patience
    )    # Train the model and catch any exceptions
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
        
        # Save the final model
        trainer.save_model(output_dir)
        logging.info(f"Model saved to {output_dir}")
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise

    # Log peak memory usage
    logging.info("Peak memory usage during training:")
    logging.info(f"  RAM: {peak_memory['ram_used_gb']:.2f} GB ({peak_memory['ram_percent']:.1f}%)")
    if torch.cuda.is_available():
        logging.info(f"  VRAM: {peak_memory['vram_used_gb']:.2f} GB ({peak_memory['vram_percent']:.1f}%)")

    # Evaluate on validation set
    logging.info("Evaluating model on validation set...")
    eval_results = trainer.evaluate()
    
    logging.info("Validation metrics:")
    for key, value in eval_results.items():
        if isinstance(value, (float, int)):
            logging.info(f"  {key}: {value:.4f}")
        elif isinstance(value, (dict, list)):
            logging.info(f"  {key}: [detailed analysis - see logs above]")
        else:
            logging.info(f"  {key}: {value}")

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
                if isinstance(value, (float, int)):
                    logging.info(f"  {key}: {value:.4f}")
                elif isinstance(value, (dict, list)):
                    logging.info(f"  {key}: [detailed analysis - see saved results]")
                else:
                    logging.info(f"  {key}: {value}")
        except Exception as e:
            logging.error(f"Testing failed with error: {str(e)}")
    
    return model, eval_results, test_metrics


def create_compute_ner_metrics(id2label):
    """
    Create a metrics computation function with access to label mappings.
    
    Args:
        id2label: Mapping from numeric IDs to string labels
        
    Returns:
        Function that computes detailed word-level NER metrics including per-label analysis
    """
    def compute_ner_metrics(pred):
        """
        Compute word-level evaluation metrics for NER task from model predictions.
        
        This function extracts word-level predictions by using only the first subtoken
        of each word (non-ignored tokens) and computes metrics at the word level,
        which is the proper way to evaluate NER models.
        
        Args:
            pred: Prediction object containing predictions and labels
                
        Returns:
            Dictionary containing overall and per-label word-level metrics, plus confusion matrix
        """
        # Extract predictions and labels
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)

        # Extract word-level predictions and labels (first subtoken only)
        word_predictions = []
        word_labels = []
        
        for prediction_seq, label_seq in zip(predictions, labels):
            # Only keep non-ignored tokens (these are first subtokens of words + special tokens)
            valid_indices = [i for i, lbl in enumerate(label_seq) if lbl != TOKEN_IGNORED_LABEL]
            
            for idx in valid_indices:
                # Skip special tokens (they should have been assigned TOKEN_IGNORED_LABEL during tokenization)
                # but double-check by ensuring the label is in our id2label mapping
                if label_seq[idx] in id2label:
                    word_predictions.append(prediction_seq[idx])
                    word_labels.append(label_seq[idx])

        # Handle edge case where no valid predictions are found
        if not word_predictions or not word_labels:
            logging.warning("No valid word-level predictions or labels found for metric calculation!")
            return {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "macro_f1": 0.0,
                "per_label_metrics": {}, "confusion_matrix": [],
                "total_words_evaluated": 0
            }

        # Calculate overall metrics using sklearn's functions with macro average
        # Macro-F1 treats all classes equally, which is better for imbalanced NER tasks
        accuracy = accuracy_score(word_labels, word_predictions)
        
        # Calculate macro-averaged metrics (unweighted mean of per-class metrics)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            word_labels,
            word_predictions,
            average='macro',     # Use macro average for balanced evaluation
            zero_division=0      # Handle zero division gracefully
        )
        
        # Also calculate weighted metrics for comparison
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            word_labels,
            word_predictions,
            average='weighted',  # Use weighted average for comparison
            zero_division=0      # Handle zero division gracefully
        )

        # Calculate per-label detailed metrics
        per_label_report = classification_report(
            word_labels, 
            word_predictions, 
            output_dict=True,
            zero_division=0,
            target_names=[id2label.get(i, f"LABEL_{i}") for i in sorted(set(word_labels + word_predictions))]
        )

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(word_labels, word_predictions)
        
        # Get unique labels for confusion matrix interpretation
        unique_labels = sorted(set(word_labels + word_predictions))
        label_names = [id2label.get(i, f"LABEL_{i}") for i in unique_labels]

        # Add small epsilon to avoid exact 1.0 values which might indicate issues
        epsilon = 1e-10
        metrics = {
            "accuracy": min(1.0 - epsilon, float(accuracy)),
            "precision": min(1.0 - epsilon, float(macro_precision)),
            "recall": min(1.0 - epsilon, float(macro_recall)),
            "macro_f1": min(1.0 - epsilon, float(macro_f1)),
            "weighted_precision": min(1.0 - epsilon, float(weighted_precision)),
            "weighted_recall": min(1.0 - epsilon, float(weighted_recall)),
            "weighted_f1": min(1.0 - epsilon, float(weighted_f1)),
            "per_label_metrics": per_label_report,
            "confusion_matrix": conf_matrix.tolist(),
            "confusion_matrix_labels": label_names,
            "total_words_evaluated": len(word_labels)
        }
        
        # Log detailed results for each label
        logging.info(f"Word-level NER evaluation on {len(word_labels)} words:")
        logging.info("Per-label NER metrics (word-level):")
        for label_name in label_names:
            if label_name in per_label_report and isinstance(per_label_report[label_name], dict):
                label_metrics = per_label_report[label_name]
                logging.info(f"  {label_name}: P={label_metrics.get('precision', 0):.3f}, "
                           f"R={label_metrics.get('recall', 0):.3f}, "
                           f"F1={label_metrics.get('f1-score', 0):.3f}, "
                           f"Support={label_metrics.get('support', 0)}")
        
        # Log overall macro and weighted metrics
        logging.info(f"Overall Word-level Macro-F1: {macro_f1:.4f}")
        logging.info(f"Overall Word-level Weighted-F1: {weighted_f1:.4f}")
        logging.info(f"Word-level Accuracy: {accuracy:.4f}")
        
        # Log confusion matrix summary
        logging.info("Word-level Confusion Matrix:")
        logging.info(f"Labels: {label_names}")
        for i, row in enumerate(conf_matrix):
            row_str = " ".join(f"{val:4d}" for val in row)
            logging.info(f"  {label_names[i]:10s} [{row_str}]")
        
        return metrics
    
    return compute_ner_metrics


def run_inference_test(model, tokenizer, test_sentence="تقع مدينة باريس في فرنسا"):
    """
    Run a sample inference test on a given sentence using word-level predictions.
    
    This function tokenizes the sentence and extracts predictions only from the first
    subtoken of each word, which aligns with our training and evaluation approach.
    
    Args:
        model: The fine-tuned NER model
        tokenizer: Tokenizer for the model
        test_sentence: Arabic test sentence to analyze
        
    Returns:
        List of (word, label) tuples for the recognized entities at word level
    """
    device = model.device
    
    # Tokenize the test sentence with word alignment
    words = test_sentence.split()  # Simple word splitting
    # First, get word alignment info from a non-tensor tokenization to use word_ids()
    tokenized_no_tensors = tokenizer(
        words,
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    # Then, prepare tensor inputs for the model and keep it as BatchEncoding (preserves methods like .to())
    inputs = tokenizer(
        words,
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    # Run model inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Process predictions at word level
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    
    # Get word IDs to align predictions with words
    # Retrieve word IDs from the non-tensor tokenization
    word_ids = tokenized_no_tensors.word_ids()
    
    # Extract word-level predictions (first subtoken only)
    word_predictions = []
    previous_word_idx = None
    
    for i, word_idx in enumerate(word_ids):
        # Skip special tokens
        if word_idx is None:
            continue
        # Only take prediction from first subtoken of each word
        elif word_idx != previous_word_idx:
            if word_idx < len(words):  # Ensure word index is valid
                word_label = model.config.id2label.get(predictions[i], "O")
                word_predictions.append((words[word_idx], word_label))
        previous_word_idx = word_idx
    
    # Log results
    logging.info(f"Test sentence: {test_sentence}")
    logging.info("Word-level predicted entities:")
    for word, label in word_predictions:
        logging.info(f"  {word}: {label}")
        
    return word_predictions


def save_benchmark_results(
    result_filepath, 
    model_name, 
    dataset_name, 
    epochs, 
    learning_rate,
    batch_size, 
    fine_tune,
    patience,
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
        patience: Early stopping patience
        metrics: Dictionary containing evaluation metrics
    """
    # Extract metrics
    accuracy = metrics.get("test_accuracy", 0.0)
    macro_f1 = metrics.get("test_macro_f1", 0.0)
    precision = metrics.get("test_precision", 0.0)
    recall = metrics.get("test_recall", 0.0)
    weighted_f1 = metrics.get("test_weighted_f1", 0.0)
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
    
    # Extract per-label and confusion matrix metrics
    per_label_metrics = metrics.get("test_per_label_metrics", {})
    confusion_matrix = metrics.get("test_confusion_matrix", [])
    confusion_matrix_labels = metrics.get("test_confusion_matrix_labels", [])
    
    # Create results structure
    results_data = {
        "configuration": {
            "model": model_name,
            "dataset": dataset_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "fine_tune": fine_tune,
            "patience": patience,
            "timestamp": datetime.now().isoformat(),
        },
        "results": {
            "accuracy": max(0.0, min(1.0, float(accuracy))),
            "macro_f1_score": max(0.0, min(1.0, float(macro_f1))),
            "precision": max(0.0, min(1.0, float(precision))),
            "recall": max(0.0, min(1.0, float(recall))),
            "weighted_f1_score": max(0.0, min(1.0, float(weighted_f1))),
            "loss": float(loss) if loss >= 0 else 0.0,
            "runtime_seconds": float(runtime),
            "samples_per_second": float(samples_per_second),
            "steps_per_second": float(steps_per_second),
        },
        "detailed_analysis": {
            "per_label_metrics": per_label_metrics,
            "confusion_matrix": confusion_matrix,
            "confusion_matrix_labels": confusion_matrix_labels,
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
        default="modernarabert",
        choices=["modernarabert", "arabert", "mbert", "arabert2", "marbert", "camel"],
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
    parser.add_argument(
        "--patience", 
        type=int, 
        default=8,
        help="Early stopping patience (number of evaluations with no improvement)"
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
        type=str,
        default= "./results/ner",
        help="Directory to save model outputs"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default= "./logs/benchmarking/ner",
        help="Directory to save logs"
    )
    
    # Test mode
    parser.add_argument(
        "--inference-test",
        action="store_true",
        help="Run a sample inference test after training"
    )
    
    return parser.parse_args()


def main():
    """Main function to run NER benchmarking"""
    # Parse arguments
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
    
    # Set up logging (level, log_file)
    setup_logging(logging.INFO, log_filepath)
    
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
    logging.info(f"  Early stopping patience: {args.patience}")
    logging.info("=" * 80)
    
    # Start benchmark timer
    start_time = time.time()
    
    try:
        # Get model and tokenizer paths
        model_path = MODEL_PATHS[args.model]["model"]
        tokenizer_path = MODEL_PATHS[args.model]["tokenizer"]
        
        # Load tokenizer
        logging.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
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
            id2label=id2label,
            output_dir=model_output_dir,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            patience=args.patience
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
            args.patience,
            test_metrics
        )
        
        # Run inference test if requested
        if args.inference_test:
            logging.info("Running sample inference test")
            run_inference_test(model, tokenizer)
        
        # Log benchmark completion
        total_time = time.time() - start_time
        logging.info(f"Benchmark completed in {total_time:.2f} seconds")
        logging.info(f"Final Macro-F1 Score: {results['results']['macro_f1_score']:.4f}")
        logging.info(f"Final Weighted-F1 Score: {results['results']['weighted_f1_score']:.4f}")
        
    except Exception as e:
        logging.error(f"Benchmark failed with error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())