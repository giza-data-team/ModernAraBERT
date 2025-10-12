"""
Text Preprocessing Module for Sentiment Analysis

This module provides utilities for Arabic text preprocessing:
- Arabic text detection
- Text chunking and windowing
- Dataset processing and splitting
- Text normalization

Original file: "text_preprocessing.py" (preprocessing functions)
Status: Logic unchanged, added missing utility functions
"""

import os
import re
import logging
from typing import List
from datasets import Dataset, DatasetDict


def is_arabic_text(text: str, threshold: float = 0.7) -> bool:
    """
    Check if text is predominantly Arabic.

    Args:
        text (str): Input text to check
        threshold (float): Minimum ratio of Arabic characters (default: 0.7)

    Returns:
        bool: True if text contains >= threshold ratio of Arabic characters
    """
    if not text or not isinstance(text, str):
        return False
    
    # Arabic Unicode range: \u0600-\u06FF
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    # Remove whitespace for character count
    non_space_chars = re.sub(r'\s', '', text)
    
    if len(non_space_chars) == 0:
        return False
    
    arabic_ratio = len(arabic_chars) / len(non_space_chars)
    return arabic_ratio >= threshold


def process_text(text: str, window_size: int) -> List[str]:
    """
    Split text into chunks based on a sliding window of words.

    Args:
        text (str): Input text to process
        window_size (int): Maximum number of words per chunk

    Returns:
        List[str]: List of text chunks
    """
    if not text or not isinstance(text, str):
        return []
    
    words = text.strip().split()
    if len(words) == 0:
        return []
    
    # If text is smaller than window, return as-is
    if len(words) <= window_size:
        return [text.strip()]
    
    # Split into chunks
    chunks = []
    for i in range(0, len(words), window_size):
        chunk = " ".join(words[i:i + window_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


def process_dataset(dataset: Dataset, window_size: int, base_dir: str):
    """
    Process a dataset by chunking texts, filtering Arabic content, and splitting into train/val/test.

    Args:
        dataset (Dataset): HuggingFace dataset with 'text', 'id', and 'label' fields
        window_size (int): Maximum words per chunk
        base_dir (str): Directory to save processed files
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
        label = example.get("label", None)
        
        # Only process Arabic text
        if is_arabic_text(text):
            chunks = process_text(text, window_size)
            for chunk in chunks:
                processed_texts.append(chunk)
                processed_ids.append(doc_id)
                processed_labels.append(label)

    print(f"Total processed chunks: {len(processed_texts)}")
    
    # Create dataset from processed data
    final_dataset = Dataset.from_dict({
        "id": processed_ids,
        "text": processed_texts,
        "label": processed_labels
    })
    
    # Split: 60% train, 20% test, 20% validation
    split_dataset = final_dataset.train_test_split(test_size=0.4, seed=42)
    test_val_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
    
    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "test": test_val_split["train"],
        "validation": test_val_split["test"]
    })
    
    # Save to TSV files
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
    
    # Log statistics
    logging.info(f"Dataset processing complete:")
    logging.info(f"  Train: {len(dataset_dict['train'])} samples")
    logging.info(f"  Test: {len(dataset_dict['test'])} samples")
    logging.info(f"  Validation: {len(dataset_dict['validation'])} samples")


if __name__ == "__main__":
    import argparse
    from datasets import load_dataset
    
    parser = argparse.ArgumentParser(description="Process SA dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset name (e.g., 'Elnagara/hard')")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed files")
    parser.add_argument("--window-size", type=int, default=8192,
                        help="Maximum words per chunk (default: 8192)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to process (default: train)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token for private datasets")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, "plain_text", split=args.split, token=args.token)
    
    print(f"Processing dataset with window size: {args.window_size}")
    process_dataset(dataset, args.window_size, args.output_dir)
    
    print("\nProcessing complete!")

