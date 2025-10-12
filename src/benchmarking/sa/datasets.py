"""
Dataset Loading Module for Sentiment Analysis Benchmarking

This module handles loading and preparation of SA datasets:
- HARD (Hate and Abusive Speech Detection)
- ASTD (Arabic Sentiment Twitter Dataset) 
- LABR (Large Arabic Book Reviews)
- AJGT (Arabic Jordanian General Tweets)

Provides functions for:
- Loading datasets from various sources (HuggingFace, CSV, XLSX)
- Processing and splitting datasets
- Tokenizing and preparing data for training

Original file: "text_preprocessing.py" (dataset loading functions)
Status: Logic unchanged, improved modularity
"""

import os
import logging
import csv
import torch
import pandas as pd
from typing import List, Tuple, Dict, Optional
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer


# Dataset configurations
DATASET_CONFIGS = {
    "hard": {
        "name": "Elnagara/hard",
        "num_labels": 2,
        "load_type": "hf",
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


def load_sentiment_dataset(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    arrow_table: Optional[object],
    num_labels: int,
    max_length: int = 512,
    dataset_type: str = "default"
) -> List[Dict[str, torch.Tensor]]:
    """
    Load and tokenize sentiment analysis dataset from a TSV file.

    Args:
        file_path (str): Path to dataset file (TSV format: id<tab>text<tab>label)
        tokenizer: HuggingFace tokenizer
        arrow_table: Arrow table (unused, kept for compatibility)
        num_labels (int): Number of classes in the dataset
        max_length (int): Maximum sequence length (default: 512)
        dataset_type (str): Dataset type for label mapping ("ajgt", "labr", "astd", "default")

    Returns:
        List of tokenized samples with input_ids, attention_mask, and labels
    """
    samples = []
    ratings = []
    
    def is_numeric(s: str) -> bool:
        """Check if string can be converted to float."""
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

            # Dataset-specific label mapping
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
                sentiment = rating
                
            elif dataset_type.lower() == "astd":
                mapping = {"obj": 0, "pos": 1, "neg": 2, "neu": 3, "neutral": 3}
                key = label.lower()
                if key in mapping:
                    sentiment = mapping[key]
                else:
                    logging.warning(f"Could not convert ASTD label for sample: {line.strip()}")
                    continue
                    
            else:  # default
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

    # Tokenize all samples
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


def prepare_astd_benchmark(data_dir: str, astd_info: Dict[str, str]):
    """
    Download and prepare ASTD dataset for benchmarking.

    Args:
        data_dir (str): Directory to save processed files
        astd_info (dict): Dataset configuration with URLs
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Load main dataset
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
    
    # Load train IDs and merge
    train_ids = pd.read_csv(astd_info["benchmark_train"], header=None, names=["id"], dtype=str)
    train_ids["id"] = train_ids["id"].str.strip().astype(int)
    train_df = pd.merge(train_ids, main_df, on="id", how="left")
    train_df.to_csv(os.path.join(data_dir, "train.txt"), sep="\t", index=False, header=False)

    # Load test IDs and merge
    test_ids = pd.read_csv(astd_info["benchmark_test"], header=None, names=["id"], dtype=str)
    test_ids["id"] = test_ids["id"].str.strip().astype(int)
    test_df = pd.merge(test_ids, main_df, on="id", how="left")
    test_df.to_csv(os.path.join(data_dir, "test.txt"), sep="\t", index=False, header=False)

    # Load validation IDs and merge
    val_ids = pd.read_csv(astd_info["benchmark_validation"], header=None, names=["id"], dtype=str)
    val_ids["id"] = val_ids["id"].str.strip().astype(int)
    val_df = pd.merge(val_ids, main_df, on="id", how="left")
    val_df.to_csv(os.path.join(data_dir, "validation.txt"), sep="\t", index=False, header=False)
    
    print(f"ASTD benchmark files prepared in {data_dir}")
    logging.info(f"ASTD: Train={len(train_df)}, Test={len(test_df)}, Val={len(val_df)}")


def prepare_labr_benchmark(data_dir: str, labr_info: Dict[str, str]):
    """
    Download and prepare LABR dataset for benchmarking.

    Args:
        data_dir (str): Directory to save processed files
        labr_info (dict): Dataset configuration with URLs
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Load main dataset
    main_df = pd.read_csv(
        labr_info["url"],
        sep="\t",
        header=None,
        names=labr_info["column_names"],
        engine="python"
    )
    
    # Convert ratings and filter
    main_df["rating"] = pd.to_numeric(main_df["rating"], errors="coerce")
    main_df = main_df.dropna(subset=["rating"])
    main_df = main_df[main_df["rating"] != 3]  # Remove neutral ratings
    main_df["label"] = main_df["rating"].apply(lambda x: 1 if x >= 4 else 0)
    
    # Create ID from index
    main_df = main_df.reset_index()  
    main_df["id"] = main_df["index"].astype(int).astype(str)
    main_df["text"] = main_df["review"].astype(str).str.strip()
    main_df = main_df[["id", "text", "label"]]
    
    print("Sample main file IDs (index-based):", main_df["id"].head(5).tolist())
    
    # Load train IDs and merge
    train_ids = pd.read_csv(labr_info["benchmark_train"], header=None, names=["id"], dtype=str)
    train_ids["id"] = train_ids["id"].astype(str).str.strip()
    print("Sample benchmark train IDs:", train_ids["id"].head(5).tolist())
    
    # Load test IDs and merge
    test_ids = pd.read_csv(labr_info["benchmark_test"], header=None, names=["id"], dtype=str)
    test_ids["id"] = test_ids["id"].astype(str).str.strip()
    print("Sample benchmark test IDs:", test_ids["id"].head(5).tolist())
    
    train_df = pd.merge(train_ids, main_df, on="id", how="inner")
    test_df = pd.merge(test_ids, main_df, on="id", how="inner")
    
    # Save to files
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    
    train_df.to_csv(train_path, sep="\t", index=False, header=False)
    test_df.to_csv(test_path, sep="\t", index=False, header=False)
    
    print(f"LABR benchmark files prepared in {data_dir}")
    print(f"Train file: {train_path} (rows: {len(train_df)})")
    print(f"Test file: {test_path} (rows: {len(test_df)})")
    logging.info(f"LABR: Train={len(train_df)}, Test={len(test_df)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare SA benchmark datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["astd", "labr"], 
                        help="Dataset to prepare")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed files")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    if args.dataset == "astd":
        prepare_astd_benchmark(args.output_dir, DATASET_CONFIGS["astd"])
    elif args.dataset == "labr":
        prepare_labr_benchmark(args.output_dir, DATASET_CONFIGS["labr"])
    
    print(f"\nDataset preparation complete!")

