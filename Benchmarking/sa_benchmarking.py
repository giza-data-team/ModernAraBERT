import re
import os
import copy
import math
import random
import json
import csv
from datetime import datetime
import logging
import argparse

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm  
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from farasa.segmenter import FarasaSegmenter
from arabert import ArabertPreprocessor


HF_TOKEN = "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"

# Set up custom cache directories for Hugging Face
CACHE_DIR = "./cache"
HF_DATASETS_CACHE = os.path.join(CACHE_DIR, "datasets")
HF_TRANSFORMERS_CACHE = os.path.join(CACHE_DIR, "transformers")

# Create cache directories if they don't exist
os.makedirs(HF_DATASETS_CACHE, exist_ok=True)
os.makedirs(HF_TRANSFORMERS_CACHE, exist_ok=True)

# Set environment variables for Hugging Face cache locations
os.environ['HF_HOME'] = CACHE_DIR
# os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE
# os.environ['TRANSFORMERS_CACHE'] = HF_TRANSFORMERS_CACHE

# Arabic Text Preprocessing with Farasa and Punctuation Fixes

# Add singleton pattern for Farasa segmenter
_farasa_segmenter = None

def get_farasa_segmenter():
    """
    Get or create a Farasa segmenter instance using singleton pattern.
    This ensures only one instance per process.
    """
    global _farasa_segmenter
    if _farasa_segmenter is None:
        model_name="bert-base-arabert"
        _farasa_segmenter = ArabertPreprocessor(model_name=model_name)
    return _farasa_segmenter

def preprocess_with_farasa(text):
    """
    Apply Farasa segmentation on input text after cleaning it of punctuation and special characters.

    Args:
        text (str): The raw Arabic text.

    Returns:
        str: The segmented and cleaned text.
    """
    text = re.sub(r'[()\[\]:«»“”‘’—_,;!?|/\\]', '', text)
    text = re.sub(r'(\-\-|\[\]|\.\.)', '', text)
    return get_farasa_segmenter().preprocess(text)

def fix_punctuation_spacing(text):
    """
    Fix spacing issues around punctuation in Arabic text.

    Args:
        text (str): Input text with potential spacing issues around punctuation.

    Returns:
        str: The text with corrected punctuation spacing.
    """
    text = re.sub(r'\s+([؟،,.!؛:])', r'\1', text)
    text = re.sub(r'([؟،,.!؛:])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def is_arabic_text(text):
    """
    Check if the given text consists only of Arabic characters (and allowed punctuation).

    Args:
        text (str): Input text.

    Returns:
        bool: True if the text matches the Arabic characters pattern, otherwise False.
    """
    arabic_pattern = re.compile(r'^[\u0600-\u06FF\s.,،؛؟!:\-–—«»“”‘’…(){}\[\]/ـ]+$')
    return bool(arabic_pattern.match(text))

def split_text_into_chunks(text, window_size):
    """
    Split text into chunks containing up to 'window_size' words each.

    Args:
        text (str): The input text.
        window_size (int): The maximum number of words per chunk.

    Returns:
        list of str: A list of text chunks.
    """
    words = text.split()
    return [" ".join(words[i:i+window_size]).strip()
            for i in range(0, len(words), window_size) if words[i:i+window_size]]

def process_text(text, window_size=8192):
    """
    Process Arabic text by applying Farasa segmentation, fixing punctuation, and splitting into chunks.

    Args:
        text (str): The raw Arabic text.
        window_size (int, optional): Maximum number of words in a chunk. Defaults to 8192.

    Returns:
        list of str: A list containing one or more processed text chunks.
    """
    processed_text = preprocess_with_farasa(text)
    processed_text = fix_punctuation_spacing(processed_text)
    words = processed_text.split()
    return [processed_text[:window_size]] if len(words) > window_size else [processed_text]

# Training routines 

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=3, learning_rate=2e-5, patience=2,
                checkpoint_path=None, continue_from_checkpoint=False, save_every=None):
    """
    Train a model on a training DataLoader with validation evaluation and early stopping.

    Uses gradient scaling (for mixed precision with CUDA) and saves checkpoints if specified.

    Args:
        model: The model to train.
        train_dataloader (DataLoader): Dataloader for training data.
        val_dataloader (DataLoader): Dataloader for validation data.
        device: Torch device (e.g., "cuda" or "cpu").
        num_epochs (int, optional): Number of epochs. Defaults to 3.
        learning_rate (float, optional): Learning rate. Defaults to 2e-5.
        patience (int, optional): Patience for early stopping. Defaults to 2.
        checkpoint_path (str, optional): Path to save model checkpoint. Defaults to None.
        continue_from_checkpoint (bool, optional): Whether to resume training from checkpoint. Defaults to False.
        save_every (int, optional): Save a checkpoint every N epochs. Defaults to None.

    Returns:
        model: The trained model (best version if early stopping occurred).
    """
    # Initialize optimizer with weight decay (no scheduler)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    if device.type == "cuda":
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    if continue_from_checkpoint and checkpoint_path is not None and os.path.exists(checkpoint_path):
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
            global_step += 1
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'lr': f'{learning_rate:.2e}'
            })
            logging.debug(f"Training Epoch {epoch+1} Batch Loss = {loss.item():.4f}, LR = {learning_rate:.2e}")
            
        avg_train_loss = epoch_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")

        _, _, _, _, _, avg_val_loss, _ = evaluate_model(model, val_dataloader, device)
        logging.info(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")
        print(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")

        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logging.info("Validation loss improved; resetting patience counter.")
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
        else:
            patience_counter += 1
            logging.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                print("Early stopping triggered.")
                break

        if save_every is not None and checkpoint_path is not None and ((epoch + 1) % save_every == 0):
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Periodic checkpoint saved at epoch {epoch+1} to {checkpoint_path}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def evaluate_model(model, eval_dataloader, device):
    """
    Evaluate a trained model on a given evaluation DataLoader.

    Computes loss, accuracy, perplexity, and returns a detailed classification report.

    Args:
        model: The model to evaluate.
        eval_dataloader (DataLoader): Dataloader for evaluation data.
        device: Torch device (e.g., "cuda" or "cpu").

    Returns:
        tuple: (accuracy, classification_report, predictions, true_labels, confidences, average_loss, perplexity)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    total_loss = 0
    total_batches = 0
    progress_bar = tqdm(eval_dataloader, desc="Evaluating", leave=False)
    
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
        else:
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
        total_loss += loss.item()
        total_batches += 1
        probs = torch.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        all_confidences.extend(max_probs.cpu().tolist())
        labels_batch = batch["labels"].cpu().tolist()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels_batch)
        progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        logging.debug(f"Evaluation Batch Loss = {loss.item():.4f}")

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    perplexity = math.exp(avg_loss)
    accuracy = accuracy_score(all_labels, all_preds)
    
    if model.config.num_labels == 4:
         target_names = ["OBJ", "POS", "NEG", "NEU"]
         report = classification_report(all_labels, all_preds, labels=[0, 1, 2, 3],
                                        target_names=target_names, zero_division=0)
    elif model.config.num_labels == 2:
         report = classification_report(all_labels, all_preds, labels=[0, 1],
                                        target_names=["Negative", "Positive"], zero_division=0)
    else:
         report = classification_report(all_labels, all_preds, zero_division=0)
    
    return accuracy, report, all_preds, all_labels, all_confidences, avg_loss, perplexity

# Dataset and Text Preprocessing routines 

def process_dataset(dataset: Dataset, window_size: int, base_dir: str, dataset_type: str = "hard"):
    """
    Process and segment texts from a dataset, then split them into train, test, and validation sets.

    This function extracts the text, applies segmentation and filtering (only if the text contains valid Arabic),
    converts labels to numeric format, and saves the processed chunks into separate TXT files for each split.

    Args:
        dataset (Dataset): Hugging Face Dataset containing examples with a "text" field.
        window_size (int): Maximum number of words per text chunk.
        base_dir (str): Directory where the split files will be saved.
        dataset_type (str): Type of dataset ('hard', 'astd', 'labr', 'ajgt')
    """
    print("Processing and segmenting texts...")
    processed_texts = []
    processed_ids = []
    processed_labels = []

    for example in dataset:
        text = example.get("text", None)
        if text is None or not isinstance(text, str) or text.strip() == "":
            continue 
        doc_id = example.get("id", None)
        label = example.get("labels", example.get("label", None))  # Try 'labels' first, fall back to 'label'
        
        # Convert label to numeric format
        label = convert_label(label, dataset_type)
        if label is None:
            continue
            
        if is_arabic_text(text):
            chunks = process_text(text, window_size)
            for chunk in chunks:
                processed_texts.append(chunk)
                processed_ids.append(doc_id)
                processed_labels.append(label)

    print(f"Total processed chunks: {len(processed_texts)}")
    final_dataset = Dataset.from_dict({
        "id": processed_ids,
        "text": processed_texts,
        "labels": processed_labels  # Changed from 'label' to 'labels'
    })
    split_dataset = final_dataset.train_test_split(test_size=0.4, seed=42)
    test_val_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "test": test_val_split["train"],
        "validation": test_val_split["test"]
    })
    print("Saving dataset to TXT files...")
    os.makedirs(base_dir, exist_ok=True)
    for split in ["train", "test", "validation"]:
        file_path = os.path.join(base_dir, f"{split}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for example in dataset_dict[split]:
                if example["labels"] is not None:  # Changed from 'label' to 'labels'
                    text_line = f"{example['id']}\t{example['text']}\t{example['labels']}"  # Changed from 'label' to 'labels'
                else:
                    text_line = f"{example['id']}\t{example['text']}"
                f.write(text_line + "\n")
        print(f"Saved {split} split to {file_path}")
    print("Dataset segmentation and splitting complete.")
    print("Files saved: train.txt, test.txt, validation.txt")

def convert_label(label_str, dataset_type):
    """
    Convert string labels to numeric labels based on dataset type.
    
    Args:
        label_str (str): String representation of the label
        dataset_type (str): Type of dataset ('hard', 'astd', 'labr', 'ajgt')
        
    Returns:
        int or None: Converted numeric label, or None if conversion fails
    """
    def is_numeric(s):
        try:
            float(s)
            return True
        except Exception:
            return False

    label_str = str(label_str).strip().replace("|", "").strip()
    if label_str == "":
        return None

    if dataset_type.lower() == "ajgt":
        if is_numeric(label_str):
            rating = int(float(label_str))
            return rating
        mapping = {"positive": 1, "negative": 0, "pos": 1, "neg": 0}
        return mapping.get(label_str.lower())

    elif dataset_type.lower() == "labr":
        if is_numeric(label_str):
            rating = int(float(label_str))
            if rating == 3:
                return None
            return 0 if rating < 3 else 1
        mapping = {"OBJ": 0, "NEG": 0, "POS": 1, "MIX": 1}
        return mapping.get(label_str.upper())

    elif dataset_type.lower() == "astd":
        mapping = {"obj": 0, "pos": 1, "neg": 2, "neu": 3, "neutral": 3}
        return mapping.get(label_str.lower())

    else:  # default/hard dataset
        if is_numeric(label_str):
            rating = int(float(label_str))
            return 0 if rating <= 1 else 1
        mapping = {"OBJ": 0, "NEUTRAL": 0, "NEG": 1, "POS": 1, "MIX": 1}
        return mapping.get(label_str.upper())

def load_sentiment_dataset(file_path, tokenizer, max_length=512, dataset_type="hard"):
    """
    Load and tokenize a sentiment dataset from a preprocessed text file.
    
    Args:
        file_path (str): Path to the preprocessed dataset file.
        tokenizer: Tokenizer to process the text.
        max_length (int, optional): Maximum tokenized sequence length.
        dataset_type (str): Type of dataset ('hard', 'astd', 'labr', 'ajgt')
    
    Returns:
        datasets.Dataset: A tokenized dataset ready for training, or None if file_path is None
    """
    if file_path is None:
        return None

    # Load the dataset using Hugging Face's datasets
    dataset = load_dataset(
        'csv',
        data_files={'data': file_path},
        delimiter='\t',
        column_names=['id', 'text', 'labels'],
        cache_dir=HF_DATASETS_CACHE
    )['data']

    # Filter examples with invalid labels
    dataset = dataset.filter(lambda x: x['labels'] is not None and str(x['labels']).strip() != "" and x['text'] is not None and str(x['text']).strip() != "")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
            return_token_type_ids=False
        )
    
    # Tokenize the dataset and set the format to PyTorch
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['id', 'text']
    )
    tokenized_dataset = tokenized_dataset.with_format("torch")

    if len(tokenized_dataset) == 0:
        logging.warning(f"No valid samples loaded from {file_path}.")
    else:
        labels = tokenized_dataset['labels']  # Changed from 'label' to 'labels'
        logging.info(f"Loaded {len(tokenized_dataset)} samples from {file_path}. Label range: min={min(labels)}, max={max(labels)}")
    
    return tokenized_dataset

def prepare_astd_benchmark(data_dir, astd_info):
    """
    Prepare benchmark files for the ASTD dataset.

    Reads the main ASTD data and benchmark split IDs, merges them to form train, test, and validation files,
    and saves these files as tab-separated text files.

    Args:
        data_dir (str): Directory in which to save the benchmark files.
        astd_info (dict): Dictionary containing URLs and file information for the ASTD dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    main_df = pd.read_csv(
        astd_info["url"],
        sep="\t",
        header=None,
        names=["text", "labels"],
        engine="python",
        quoting=csv.QUOTE_NONE
    )
    main_df["id"] = main_df.index.astype(str)
    
    
    # Apply Farasa segmentation to text
    main_df["text"] = main_df["text"].apply(lambda x: process_text(x)[0] if isinstance(x, str) else x)
    # Convert labels before saving
    main_df["labels"] = main_df["labels"].apply(lambda x: convert_label(x, "astd"))
    # Filter out rows with None labels
    main_df = main_df.dropna(subset=["labels"])
    # Ensure label is integer
    main_df["labels"] = main_df["labels"].astype(int)
    
    main_df = main_df[["id", "text", "labels"]]
    
    train_ids = pd.read_csv(astd_info["benchmark_train"], header=None, names=["id"], dtype=str)
    train_ids["id"] = train_ids["id"].str.strip()
    train_df = pd.merge(train_ids, main_df, on="id", how="left")
    train_df.to_csv(os.path.join(data_dir, "train.txt"), sep="\t", index=False, header=False)

    test_ids = pd.read_csv(astd_info["benchmark_test"], header=None, names=["id"], dtype=str)
    test_ids["id"] = test_ids["id"].str.strip()
    test_df = pd.merge(test_ids, main_df, on="id", how="left")
    test_df.to_csv(os.path.join(data_dir, "test.txt"), sep="\t", index=False, header=False)

    val_ids = pd.read_csv(astd_info["benchmark_validation"], header=None, names=["id"], dtype=str)
    val_ids["id"] = val_ids["id"].str.strip()
    val_df = pd.merge(val_ids, main_df, on="id", how="left")
    val_df.to_csv(os.path.join(data_dir, "validation.txt"), sep="\t", index=False, header=False)
    print(f"ASTD benchmark files prepared in {data_dir}")

def prepare_labr_benchmark(data_dir, labr_info):
    """
    Prepare benchmark files for the LABR dataset.

    Loads the LABR dataset from a given URL, cleans and processes the data, 
    merges benchmark IDs with the main dataset, and saves train and test files.

    Args:
        data_dir (str): Directory to save the benchmark files.
        labr_info (dict): Dictionary containing URLs and column names for LABR.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    main_df = pd.read_csv(
        labr_info["url"],
        sep="\t",
        header=None,
        names=labr_info["column_names"],
        engine="python"
    )
    
    # Convert labels using the convert_label function
    main_df["rating"] = main_df["rating"].apply(lambda x: convert_label(x, "labr"))
    main_df = main_df.dropna(subset=["rating"])
    main_df["labels"] = main_df["rating"].astype(int)
    
    main_df = main_df.reset_index()  
    main_df["id"] = main_df["index"].astype(int).astype(str)
    main_df["text"] = main_df["review"].astype(str).str.strip()
        
    # Apply Farasa segmentation to text
    main_df["text"] = main_df["text"].apply(lambda x: process_text(x)[0] if isinstance(x, str) else x)

    main_df = main_df[["id", "text", "labels"]]
    
    print("Sample main file IDs (index-based):", main_df["id"].head(5).tolist())
    
    train_ids = pd.read_csv(labr_info["benchmark_train"], header=None, names=["id"], dtype=str)
    train_ids["id"] = train_ids["id"].astype(str).str.strip()
    print("Sample benchmark train IDs:", train_ids["id"].head(5).tolist())
    
    test_ids = pd.read_csv(labr_info["benchmark_test"], header=None, names=["id"], dtype=str)
    test_ids["id"] = test_ids["id"].astype(str).str.strip()
    print("Sample benchmark test IDs:", test_ids["id"].head(5).tolist())
    
    train_df = pd.merge(train_ids, main_df, on="id", how="inner")
    test_df = pd.merge(test_ids, main_df, on="id", how="inner")
    
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    
    train_df.to_csv(train_path, sep="\t", index=False, header=False)
    test_df.to_csv(test_path, sep="\t", index=False, header=False)
    
    print(f"LABR benchmark files prepared in {data_dir}")
    print(f"Train file: {train_path} (rows: {len(train_df)})")
    print(f"Test file: {test_path} (rows: {len(test_df)})")

# Generate a unique log filename based on model, epochs, patience, and timestamp
def get_log_filename(model_name, epochs, patience, dataset_name):
    """
    Generate a unique log filename based on model name, epoch count, patience, dataset, and current timestamp.

    Args:
        model_name (str): The name of the model.
        epochs (int): Number of epochs.
        patience (int): Early stopping patience value.
        dataset_name (str): The selected dataset name.

    Returns:
        str: The generated log filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"SA_Benchmark_{model_name}_{dataset_name}_{epochs}ep_p{patience}_{timestamp}.log"

datasets_dict = {
    "hard": {
        "name": "Elnagara/hard",
        "num_labels": 2,
        "load_type": "hf",
        "token": HF_TOKEN
    },
    "astd": {
        "name": "ASTD",
        "num_labels": 4,
        "load_type": "csv",
        "url": "https://raw.githubusercontent.com/mahmoudnabil/ASTD/master/data/Tweets.txt?raw=true",
        "benchmark_train": "https://raw.githubusercontent.com/mahmoudnabil/ASTD/master/data/4class-balanced-train.txt?raw=true",
        "benchmark_test": "https://raw.githubusercontent.com/mahmoudnabil/ASTD/master/data/4class-balanced-test.txt?raw=true",
        "benchmark_validation": "https://raw.githubusercontent.com/mahmoudnabil/ASTD/master/data/4class-balanced-validation.txt?raw=true",
        "column_names": ["text", "label"]
    },
    "labr": {
        "name": "LABR",
        "num_labels": 2,
        "load_type": "csv",
        "url": "https://raw.githubusercontent.com/mohamedadaly/LABR/master/data/reviews.tsv?raw=true",
        "benchmark_train": "https://raw.githubusercontent.com/mohamedadaly/LABR/master/data/2class-unbalanced-train.txt?raw=true",
        "benchmark_test": "https://raw.githubusercontent.com/mohamedadaly/LABR/master/data/2class-unbalanced-test.txt?raw=true",
        "column_names": ["rating", "review id", "user_id", "book_id", "review"]
    },
    "ajgt": {
        "name": "AJGT",
        "num_labels": 2,
        "load_type": "xlsx",
        "url": "https://github.com/komari6/Arabic-twitter-corpus-AJGT/blob/master/AJGT.xlsx?raw=true",
        "column_names": ["text", "label"]
    }
}

models = {
    "modernbert": "./model_checkpoints/checkpoint_step_13000/",
    "arabert": "aubmindlab/bert-base-arabert",
    "distilbert": "shahendaadel211/arabic-distilbert-model"
}

def create_dataloader(dataset, batch_size, num_workers, shuffle=False):
    """
    Create a DataLoader with standard configuration.
    
    Args:
        dataset: The dataset to load
        batch_size (int): Batch size
        num_workers (int): Number of worker threads
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
    
    Returns:
        DataLoader: Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def prepare_dataset(dataset_name, data_dir, dataset_info, preprocess_flag=False):
    """
    Prepare dataset files if they don't exist and preprocess_flag is True.
    If preprocess_flag is False, only load from local storage and throw error if files don't exist.
    
    Args:
        dataset_name (str): Name of the dataset
        data_dir (str): Directory to store dataset files
        dataset_info (dict): Dataset configuration
        preprocess_flag (bool): Whether to prepare datasets (True) or only load from local (False)
        
    Raises:
        FileNotFoundError: If preprocess_flag is False and dataset files don't exist
    """
    os.makedirs(data_dir, exist_ok=True)
    
    files_to_check = [
        os.path.join(data_dir, "train.txt"),
        os.path.join(data_dir, "test.txt")
    ]
    if dataset_name != "labr":
        files_to_check.append(os.path.join(data_dir, "validation.txt"))
    
    all_files_exist = all(os.path.exists(f) for f in files_to_check)
    
    # If all files exist, no need for preparation
    if all_files_exist:
        print(f"Dataset {dataset_name} files found in {data_dir}")
        return
    
    # If files don't exist and preprocess_flag is False, raise error
    if not preprocess_flag:
        missing_files = [f for f in files_to_check if not os.path.exists(f)]
        raise FileNotFoundError(
            f"Dataset {dataset_name} files not found in {data_dir}. "
            f"Missing files: {', '.join([os.path.basename(f) for f in missing_files])}. "
            "Please run with --preprocess-flag to prepare the dataset."
        )
    
    # Only proceed with preparation if preprocess_flag is True
    print(f"Preparing {dataset_name} dataset...")
    
    if dataset_name == "hard":
        dataset = load_dataset(
            "Elnagara/hard",
            "plain_text",
            split="train",
            token=HF_TOKEN,
            cache_dir=HF_DATASETS_CACHE
        )
        process_dataset(dataset, window_size=8192, base_dir=data_dir, dataset_type=dataset_name)
    elif dataset_name == "astd":
        prepare_astd_benchmark(data_dir, dataset_info)
    elif dataset_name == "labr":
        prepare_labr_benchmark(data_dir, dataset_info)
    elif dataset_name == "ajgt":
        df = pd.read_excel(dataset_info["url"], engine="openpyxl")
        df = df.iloc[:, 1:3] if len(df.columns) == 3 else df.iloc[:, :2]
        df.columns = dataset_info["column_names"]
        dataset = Dataset.from_pandas(df)
        process_dataset(dataset, window_size=8192, base_dir=data_dir, dataset_type=dataset_name)

def configure_model(model_name, model_path, num_labels, device, freeze=False):
    """
    Configure the model with the specified settings.
    
    Args:
        model_name (str): Name of the model
        model_path (str): Path to model files
        num_labels (int): Number of output labels
        device: The device to put the model on
        freeze (bool): Whether to freeze model parameters
    
    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer_path = model_path if model_name != "modernbert" else MODERN_BERT_TOKENIZER_PATH
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        # token=HF_TOKEN,
        cache_dir=HF_TRANSFORMERS_CACHE
    )
    
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=num_labels,
        # token=HF_TOKEN,
        cache_dir=HF_TRANSFORMERS_CACHE
    )
    model = AutoModelForSequenceClassification.from_config(config)
    model.classifier = nn.Linear(model.config.hidden_size, num_labels)
    
    if freeze:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
                
    model.to(device)
    return model, tokenizer


# Main Benchmarking Script 



"""
Arguments:
- --model-name: Choose between 'modernbert' or 'arabert'.
- --dataset: Select dataset: 'hard', 'astd', 'labr', or 'ajgt'.
- --benchmark: Use the test set for evaluation instead of validation.
- --max-size: Max sequence length for tokenization.
- --batch-size: Number of samples per batch.
- --epochs: Total number of training epochs.
- --learning-rate: Learning rate for training.
- --split-ratio: Train/test/validation split (comma-separated).
- --num-workers: Number of worker threads for data loading.
- --freeze: Freeze model layers except the classifier head.
- --checkpoint: Path to save/load model checkpoint.
- --continue-from-checkpoint: Resume training from a saved checkpoint.
- --preprocess-flag: Skip preprocessing if already done.
- --save-every: Save checkpoint every N epochs.
- --patience: Patience for early stopping if validation loss doesn't improve.
"""

parser = argparse.ArgumentParser("Sentiment Analysis Benchmarking")
parser.add_argument("--model-name", dest="model_name", type=str, default='modernbert',
                    choices=['modernbert', 'arabert', 'distilbert'])
parser.add_argument("--dataset", dest="dataset_name", type=str, default='hard',
                    choices=['hard', 'astd', 'labr', 'ajgt'])
parser.add_argument("--benchmark", dest="benchmark", action='store_true',
                    help="Use test set for evaluation instead of validation set")
parser.add_argument("--max-size", dest="max_size", type=int, default=512)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=16)
parser.add_argument("--epochs", dest="epochs", type=int, default=50)
parser.add_argument("--learning-rate", dest="learning_rate",
                    type=float, default=2e-5)
parser.add_argument("--split-ratio", dest="split_ratio", type=str, default="0.6,0.2,0.2",
                    help="Train, Test, Validation split ratios, comma separated")
parser.add_argument("--num-workers", dest="num_workers", type=int, default=2)
parser.add_argument("--freeze", dest="freeze", type=bool, default=True,
                    help="Freeze model parameters except classifier head")
parser.add_argument("--checkpoint", dest="checkpoint", type=str, default=None,
                    help="Path to save/load model checkpoint")
parser.add_argument("--continue-from-checkpoint", dest="continue_from_checkpoint", action='store_true',
                    help="Flag to load from a saved checkpoint and continue training")
parser.add_argument("--preprocess-flag", dest="preprocess_flag", type=bool, default=True,
                    help="If True, preprocess and prepare datasets. If False, only load from local storage.")
parser.add_argument("--save-every", dest="save_every", type=int, default=None,
                    help="Save checkpoint every N epochs (if provided)")
parser.add_argument("--patience", dest="patience", type=int, default=5,
                    help="Early stopping patience value")
args = parser.parse_args()

# Set patience value 
PATIENCE = args.patience
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename
log_filename = get_log_filename(args.model_name, args.epochs, PATIENCE, args.dataset_name)
log_filepath = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_filepath)],
    force=True,
)
# logging.info(f"Starting benchmarking for model {args.model_name} on dataset {args.dataset_name}")
# logging.info(f"Configuration: epochs={args.epochs}, patience={PATIENCE}, batch_size={args.batch_size}")
# print(f"Logging to {log_filepath}")

PARENT_PATH = "./data"

MODERN_BERT_TOKENIZER_PATH = "./Tokenizer"

DATA_DIR = f"{PARENT_PATH}/{args.dataset_name.lower()}"

if args.dataset_name.lower() == "labr":
    TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
    VAL_FILE = None
    TEST_FILE = os.path.join(DATA_DIR, "test.txt")
else:
    TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
    VAL_FILE = os.path.join(DATA_DIR, "validation.txt")
    TEST_FILE = os.path.join(DATA_DIR, "test.txt")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    chosen_dataset = args.dataset_name.lower()
    try:
        prepare_dataset(chosen_dataset, DATA_DIR, datasets_dict[chosen_dataset], args.preprocess_flag)

        EVAL_FILE = TEST_FILE if args.benchmark else (
            VAL_FILE if chosen_dataset != "labr" else TEST_FILE)

        def main():
            """
            Main function for running sentiment analysis benchmarking.

            Loads model and tokenizer, processes datasets, trains the model, evaluates its performance,
            and saves both the final model and the benchmarking results.

            This function orchestrates the complete workflow:
                - Device selection and logging.
                - Loading data and preparing dataloaders.
                - Training the model with early stopping and checkpointing.
                - Evaluating model performance and saving results.
            """
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")
            print(f"Using device: {device}")

            benchmark_results = {}
            model_predictions = {}
            model_confidences = {}
            model_true_labels = None

            model_name = args.model_name
            model_path = models.get(model_name)
            logging.info(f"Benchmarking model: {model_name}")
            print(f"\nBenchmarking model: {model_name}")

            model, tokenizer = configure_model(
                model_name,
                model_path,
                datasets_dict[args.dataset_name]['num_labels'],
                device,
                args.freeze
            )

            train_dataset = load_sentiment_dataset(
                TRAIN_FILE,
                tokenizer,
                max_length=args.max_size,
                dataset_type=args.dataset_name
            )
            if len(train_dataset) == 0:
                logging.error(f"No training samples loaded for model {model_name}.")
                raise Exception("No training samples.")


            # Handle validation dataset
            if VAL_FILE is None:
                logging.info("No validation file specified, using a portion of training data for validation")
                # Split training dataset into train and validation
                train_val_split = train_dataset.train_test_split(test_size=0.2, seed=42)
                val_dataset = train_val_split['test']
                # Update train_dataset to be smaller
                train_dataset = train_val_split['train']
            else:
                val_dataset = load_sentiment_dataset(
                    VAL_FILE,
                    tokenizer,
                    max_length=args.max_size,
                    dataset_type=args.dataset_name
                )
                if len(val_dataset) == 0:
                    logging.error(f"No validation samples loaded for model {model_name}.")
                    raise Exception("No validation samples.")

            train_dataloader = create_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
            val_dataloader = create_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

            logging.info(f"Starting training for model: {model_name}")
            model_trained = train_model(
                model,
                train_dataloader,
                val_dataloader,
                device,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=PATIENCE,
                checkpoint_path=args.checkpoint,
                continue_from_checkpoint=args.continue_from_checkpoint,
                save_every=args.save_every
            )

            test_dataset = load_sentiment_dataset(
                TEST_FILE,
                tokenizer,
                max_length=args.max_size,
                dataset_type=args.dataset_name
            )
            
            test_dataloader = create_dataloader(test_dataset, args.batch_size, args.num_workers, shuffle=False)

            accuracy, report, preds, true_labels, confidences, avg_eval_loss, perplexity = evaluate_model(
                model_trained, test_dataloader, device)
            benchmark_results[model_name] = {
                "accuracy": accuracy,
                "report": report,
                "avg_eval_loss": avg_eval_loss,
                "perplexity": perplexity
            }
            model_predictions[model_name] = preds
            model_confidences[model_name] = confidences
            if model_true_labels is None:
                model_true_labels = true_labels

            logging.info(f"{model_name} Evaluation Accuracy: {accuracy:.4f}")
            logging.info(f"{model_name} Classification Report:\n{report}")
            logging.info(
                f"{model_name} Average Eval Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.4f}")
            print(f"{model_name} Evaluation Accuracy: {accuracy:.4f}")
            print(f"{model_name} Classification Report:")
            print(report)
            print(
                f"{model_name} Average Evaluation Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.4f}")

            print(
                f"\nComparison of ground truth, predictions, and confidences for {model_name}:")
            print(f"{'Index':<6}{'Ground Truth':<15}{'Prediction':<15}{'Confidence':<12}")
            for i in range(min(10, len(true_labels))):
                print(
                    f"{i:<6}{true_labels[i]:<15}{preds[i]:<15}{confidences[i]:<12.4f}")
            print("-" * 40)

            if args.checkpoint is not None:
                torch.save(model_trained.state_dict(), args.checkpoint)
                tokenizer.save_pretrained(os.path.dirname(args.checkpoint))
                logging.info(
                    f"Final model and tokenizer saved to {args.checkpoint} and its directory.")

            torch.cuda.empty_cache()

            # Save benchmark results to a JSON file with the same base name as the log
            result_filepath = log_filepath.replace('.log', '_results.json')
            with open(result_filepath, 'w') as f:
                json.dump({
                    "configuration": {
                        "model": model_name,
                        "dataset": args.dataset_name,
                        "epochs": args.epochs,
                        "patience": PATIENCE,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "max_sequence_length": args.max_size,
                        "freeze_layers": args.freeze
                    },
                    "results": {
                        model_name: {
                            "accuracy": float(accuracy),
                            "avg_eval_loss": float(avg_eval_loss),
                            "perplexity": float(perplexity)
                        }
                    }
                }, f, indent=2)
            logging.info(f"Results saved to {result_filepath}")
            print(f"Detailed results saved to {result_filepath}")

            print("\nBenchmarking Complete. Summary of Results:")
            for m_name, metrics in benchmark_results.items():
                print(f"{m_name}: Accuracy = {metrics['accuracy']:.4f} | Perplexity = {metrics['perplexity']:.4f}")

        main()
    except FileNotFoundError as e:
        logging.error(f"Dataset Error: {str(e)}")
        print(f"ERROR: {str(e)}")
        print("Make sure to run with --preprocess-flag to prepare the dataset files first.")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"ERROR: An unexpected error occurred: {str(e)}")
