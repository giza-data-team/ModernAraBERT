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

# Arabic Text Preprocessing with Farasa and Punctuation Fixes

# Initialize the FarasaSegmenter (interactive mode)
farasa_segmenter = FarasaSegmenter(interactive=True)

def preprocess_with_farasa(text):
    """Apply Farasa segmentation and clean text."""
    text = re.sub(r'[()\[\]:«»“”‘’—_,;!?|/\\]', '', text)
    text = re.sub(r'(\-\-|\[\]|\.\.)', '', text)
    return farasa_segmenter.segment(text)

def fix_punctuation_spacing(text):
    """Fix Arabic punctuation spacing issues."""
    text = re.sub(r'\s+([؟،,.!؛:])', r'\1', text)
    text = re.sub(r'([؟،,.!؛:])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def is_arabic_text(text):
    """Check if text contains Arabic characters."""
    arabic_pattern = re.compile(r'^[\u0600-\u06FF\s.,،؛؟!:\-–—«»“”‘’…(){}\[\]/ـ]+$')
    return bool(arabic_pattern.match(text))

def split_text_into_chunks(text, window_size):
    """Split text into chunks of max window_size words."""
    words = text.split()
    return [" ".join(words[i:i+window_size]).strip()
            for i in range(0, len(words), window_size) if words[i:i+window_size]]

def process_text(text, window_size=8192):
    """Apply full text processing: segmentation, punctuation fixes, and chunking."""
    processed_text = preprocess_with_farasa(text)
    processed_text = fix_punctuation_spacing(processed_text)
    words = processed_text.split()
    return split_text_into_chunks(processed_text, window_size) if len(words) > window_size else [processed_text]

# Training routines 

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=3, learning_rate=2e-5, patience=2,
                checkpoint_path=None, continue_from_checkpoint=False, save_every=None):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if continue_from_checkpoint and checkpoint_path is not None and os.path.exists(checkpoint_path):
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
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
            epoch_loss += loss.item()
            logging.info(f"Training Epoch {epoch+1} Batch Loss = {loss.item():.4f}")
            
        avg_train_loss = epoch_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")

        _, _, _, _, _, avg_val_loss, _ = evaluate_model(model, val_dataloader, device)
        logging.info(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")
        print(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
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
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    total_loss = 0
    total_batches = 0
    
    for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
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
        logging.info(f"Evaluation Batch Loss = {loss.item():.4f}")

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

def process_dataset(dataset: Dataset, window_size: int, base_dir: str):
    print("Processing and segmenting texts...")
    processed_texts = []
    processed_ids = []
    processed_labels = []

    for example in dataset:
        text = example.get("text", None)
        if text is None or not isinstance(text, str) or text.strip() == "":
            continue 
        doc_id = example.get("id", None)
        label = example.get("label", None)
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
        "label": processed_labels
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
                if example["label"] is not None:
                    text_line = f"{example['id']}\t{example['text']}\t{example['label']}"
                else:
                    text_line = f"{example['id']}\t{example['text']}"
                f.write(text_line + "\n")
        print(f"Saved {split} split to {file_path}")
    print("Dataset segmentation and splitting complete.")
    print("Files saved: train.txt, test.txt, validation.txt")

def load_sentiment_dataset(file_path, tokenizer, arrow_table, num_labels, max_length=512, dataset_type="default"):
    samples = []
    ratings = []
    
    def is_numeric(s):
        try:
            float(s)
            return True
        except Exception:
            return False

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", maxsplit=2)
            if len(parts) != 3:
                logging.warning(f"Line skipped due to insufficient parts: {line.strip()}")
                continue
            _, text, label = parts
            if text is None or not isinstance(text, str) or text.strip() == "":
                continue
            label = label.strip().replace("|", "").strip()
            if label == "":
                continue

            if dataset_type.lower() == "ajgt":
                if is_numeric(label):
                    rating = int(float(label))
                else:
                    mapping = {"positive": 1, "negative": 0, "pos": 1, "neg": 0}
                    key = label.lower()
                    if key in mapping:
                        rating = mapping[key]
                    else:
                        logging.warning(f"Could not convert AJGT label for sample: {line.strip()}")
                        continue
                sentiment = rating  
            elif dataset_type.lower() == "labr":
                if is_numeric(label):
                    rating = int(float(label))
                else:
                    mapping = {"OBJ": 0, "NEG": 1, "POS": 1, "MIX": 1}
                    key = label.upper()
                    if key in mapping:
                        rating = mapping[key]
                    else:
                        logging.warning(f"Could not convert LABR label for sample: {line.strip()} | Error: {label}")
                        continue
                if rating == 3:
                    continue 
                sentiment = 0 if rating < 3 else 1
            elif dataset_type.lower() == "astd":
                mapping = {"obj": 0, "pos": 1, "neg": 2, "neu": 3, "neutral": 3}
                key = label.lower()
                if key in mapping:
                    sentiment = mapping[key]
                else:
                    logging.warning(f"Could not convert ASTD label for sample: {line.strip()}")
                    continue
            else:
                if is_numeric(label):
                    rating = int(float(label))
                else:
                    mapping = {"OBJ": 0, "NEUTRAL": 0, "NEG": 1, "POS": 1, "MIX": 1}
                    key = label.upper()
                    if key in mapping:
                        rating = mapping[key]
                    else:
                        logging.warning(f"Could not convert label for sample: {line.strip()}")
                        continue
                sentiment = 0 if rating <= 1 else 1

            if 0 <= sentiment < num_labels:
                samples.append((text, sentiment))
                ratings.append(sentiment)
            else:
                logging.warning(f"Invalid sentiment {sentiment} for text: {text[:30]}...")

    if ratings:
        logging.info(f"Loaded {len(samples)} samples from {file_path}. Label range: min={min(ratings)}, max={max(ratings)}")
    else:
        logging.warning(f"No valid samples loaded from {file_path}.")

    tokenized_samples = []
    for text, label in samples:
        if isinstance(text, list):
            text = " ".join(text)
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long).squeeze()
        tokenized_samples.append(item)

    return tokenized_samples

def prepare_astd_benchmark(data_dir, astd_info):
    os.makedirs(data_dir, exist_ok=True)
    main_df = pd.read_csv(
        astd_info["url"],
        sep="\t",
        header=None,
        names=["text", "label"],
        engine="python",
        quoting=csv.QUOTE_NONE
    )
    main_df["id"] = main_df.index.astype(str)
    main_df = main_df[["id", "text", "label"]]
    
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
    os.makedirs(data_dir, exist_ok=True)
    
    main_df = pd.read_csv(
        labr_info["url"],
        sep="\t",
        header=None,
        names=labr_info["column_names"],
        engine="python"
    )
    
    main_df["rating"] = pd.to_numeric(main_df["rating"], errors="coerce")
    main_df = main_df.dropna(subset=["rating"])
    main_df = main_df[main_df["rating"] != 3]
    main_df["label"] = main_df["rating"].apply(lambda x: 1 if x >= 4 else 0)
    
    main_df = main_df.reset_index()  
    main_df["id"] = main_df["index"].astype(int).astype(str)
    main_df["text"] = main_df["review"].astype(str).str.strip()
    main_df = main_df[["id", "text", "label"]]
    
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

# Main Benchmarking Script 

HF_TOKEN = "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"

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
- --use-streaming: Use streaming dataset loading (no in-memory splitting).
- --patience: Patience for early stopping if validation loss doesn't improve.
"""

parser = argparse.ArgumentParser("Sentiment Analysis Benchmarking")
parser.add_argument("--model-name", dest="model_name", type=str, default='modernbert',
                    choices=['modernbert', 'arabert'])
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
parser.add_argument("--freeze", dest="freeze", action='store_true',
                    help="Freeze model parameters except classifier head")
parser.add_argument("--checkpoint", dest="checkpoint", type=str, default=None,
                    help="Path to save/load model checkpoint")
parser.add_argument("--continue-from-checkpoint", dest="continue_from_checkpoint", action='store_true',
                    help="Flag to load from a saved checkpoint and continue training")
parser.add_argument("--preprocess-flag", dest="preprocess_flag", action='store_true',
                    help="If True, skip dataset preprocessing (assumes preprocessing is already done)")
parser.add_argument("--save-every", dest="save_every", type=int, default=None,
                    help="Save checkpoint every N epochs (if provided)")
parser.add_argument("--use-streaming", dest="use_streaming", action='store_true',
                    help="Use streaming mode for dataset (splitting not supported in streaming mode)")
parser.add_argument("--patience", dest="patience", type=int, default=5,
                    help="Early stopping patience value")
args = parser.parse_args()

# Generate a unique log filename based on model, epochs, patience, and timestamp
def get_log_filename(model_name, epochs, patience, dataset_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"SA_Benchmark_{model_name}_{dataset_name}_{epochs}ep_p{patience}_{timestamp}.log"

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
logging.info(f"Starting benchmarking for model {args.model_name} on dataset {args.dataset_name}")
logging.info(f"Configuration: epochs={args.epochs}, patience={PATIENCE}, batch_size={args.batch_size}")
print(f"Logging to {log_filepath}")

PARENT_PATH = "./data"

MODERN_BERT_TOKENIZER_PATH = "./Tokenizer"

if args.dataset_name.lower() == "hard":
    DATA_DIR = f"{PARENT_PATH}/hard"
elif args.dataset_name.lower() in ["astd", "labr", "ajgt"]:
    DATA_DIR = f"{PARENT_PATH}/{args.dataset_name.lower()}"
else:
    DATA_DIR = f"{PARENT_PATH}/{args.dataset_name.lower()}"

if args.dataset_name.lower() == "labr":
    TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
    TEST_FILE = os.path.join(DATA_DIR, "test.txt")
    EVAL_FILE = TEST_FILE
else:
    TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
    VAL_FILE = os.path.join(DATA_DIR, "validation.txt")
    TEST_FILE = os.path.join(DATA_DIR, "test.txt")
    EVAL_FILE = TEST_FILE if args.benchmark else VAL_FILE

datasets_dict = {
    "hard": {
        "name": "Elnagara/hard",
        "num_labels": 2,
        "load_type": "hf",
        "token": HF_TOKEN
    },
    "astd": {
        "name": "ASTD",
        "num_labels": 2,
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
    "arabert": "aubmindlab/bert-base-arabert"
}

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    chosen_dataset = args.dataset_name.lower()
    if chosen_dataset == "hard":
        DATA_DIR = f"{PARENT_PATH}/hard"
        dataset = load_dataset("Elnagara/hard", "plain_text",
                               split="train", token=HF_TOKEN)
        TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
        VAL_FILE = os.path.join(DATA_DIR, "validation.txt")
        TEST_FILE = os.path.join(DATA_DIR, "test.txt")
        if not (os.path.exists(TRAIN_FILE) and os.path.exists(VAL_FILE) and os.path.exists(TEST_FILE)):
            process_dataset(dataset, window_size=8192, base_dir=DATA_DIR)
    elif chosen_dataset == "astd":
        DATA_DIR = f"{PARENT_PATH}/astd"
        TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
        VAL_FILE = os.path.join(DATA_DIR, "validation.txt")
        TEST_FILE = os.path.join(DATA_DIR, "test.txt")
        if not (os.path.exists(TRAIN_FILE) and os.path.exists(VAL_FILE) and os.path.exists(TEST_FILE)):
            prepare_astd_benchmark(DATA_DIR, datasets_dict["astd"])
        dataset = None
    elif chosen_dataset == "labr":
        DATA_DIR = f"{PARENT_PATH}/labr"
        prepare_labr_benchmark(DATA_DIR, datasets_dict["labr"])
        TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
        TEST_FILE = os.path.join(DATA_DIR, "test.txt")
        EVAL_FILE = TEST_FILE
        dataset = None
    elif chosen_dataset == "ajgt":
        DATA_DIR = f"{PARENT_PATH}/ajgt"
        df = pd.read_excel(datasets_dict["ajgt"]["url"], engine="openpyxl")
        if len(df.columns) == 3:
            df = df.iloc[:, 1:3]
        else:
            df = df.iloc[:, :2]
        df.columns = datasets_dict["ajgt"]["column_names"]
        ds = Dataset.from_pandas(df)
        TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
        VAL_FILE = os.path.join(DATA_DIR, "validation.txt")
        TEST_FILE = os.path.join(DATA_DIR, "test.txt")
        if not (os.path.exists(TRAIN_FILE) and os.path.exists(VAL_FILE) and os.path.exists(TEST_FILE)):
            process_dataset(ds, window_size=8192, base_dir=DATA_DIR)
        dataset = ds
    else:
        raise ValueError("Unknown dataset selected.")

    if chosen_dataset not in ["astd", "labr"] and not args.use_streaming:
        if not args.preprocess_flag:
            process_dataset(dataset, window_size=8192, base_dir=DATA_DIR)
    elif args.use_streaming:
        print("Streaming mode enabled. Skipping in-memory processing and splitting.")

    EVAL_FILE = TEST_FILE if args.benchmark else (
        VAL_FILE if chosen_dataset != "labr" else TEST_FILE)

    def main():
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

        tokenizer_path = model_path if model_name == "arabert" else MODERN_BERT_TOKENIZER_PATH

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, token=HF_TOKEN)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, token=HF_TOKEN)

        config = AutoConfig.from_pretrained(
            model_path, num_labels=datasets_dict[args.dataset_name]['num_labels'], token=HF_TOKEN)
        hidden_size = model.config.hidden_size
        model.classifier = nn.Linear(
            hidden_size, datasets_dict[args.dataset_name]['num_labels'])

        if args.freeze:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        model.to(device)

        extra_kwargs = {}
        if args.dataset_name.lower() == "labr":
            extra_kwargs["dataset_type"] = "labr"
        elif args.dataset_name.lower() == "ajgt":
            extra_kwargs["dataset_type"] = "ajgt"
        else:
            extra_kwargs["dataset_type"] = "default"

        train_samples = load_sentiment_dataset(
            TRAIN_FILE,
            tokenizer,
            None,
            num_labels=datasets_dict[args.dataset_name]['num_labels'],
            max_length=args.max_size,
            dataset_type=extra_kwargs.get("dataset_type", "default")
        )
        if len(train_samples) == 0:
            logging.error(
                f"No training samples loaded for model {model_name}.")
            raise Exception("No training samples.")
        train_dataloader = DataLoader(
            train_samples,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        val_samples = load_sentiment_dataset(
            EVAL_FILE,
            tokenizer,
            None,
            num_labels=datasets_dict[args.dataset_name]['num_labels'],
            max_length=args.max_size,
            dataset_type=extra_kwargs.get("dataset_type", "default")
        )
        if len(val_samples) == 0:
            if args.dataset_name.lower() == "labr":
                logging.warning(
                    "No validation samples loaded for LABR dataset; using training samples as validation.")
                val_samples = train_samples
            else:
                logging.error(
                    f"No validation samples loaded for model {model_name}.")
                raise Exception("No validation samples.")
        val_dataloader = DataLoader(
            val_samples,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

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

        accuracy, report, preds, true_labels, confidences, avg_eval_loss, perplexity = evaluate_model(
            model_trained, val_dataloader, device)
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
