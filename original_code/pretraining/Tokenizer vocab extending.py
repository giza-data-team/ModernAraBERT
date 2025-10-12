"""
The following modifications have been implemented to improve the vocabulary augmentation process:

1. **Refined Vocabulary Analysis**:
   - Introduced a new function, `analyze_vocab_distribution`, to better analyze and filter the vocabulary.
   - Uses a more precise Arabic regex pattern (`[\u0621-\u063A\u0641-\u064A]+`) to capture valid Arabic words.
   - Counts both:
     - **Base words** (after removing affixes)
     - **Full words** (keeping affixes intact)

2. **Vocabulary Size Capping**:
   - The vocabulary is now capped at a maximum of **80K tokens**.
   - After collecting tokens (from both base words and full words) and adding common affixes, the vocabulary is trimmed if its size exceeds the 80K threshold.
   - This ensures the final vocabulary remains manageable and closer to the expected size (e.g., AraBERT has ~60K tokens).

3. **Additional Token Handling**:
   - Added special tokens (e.g., the `+` token) for segmentation.
   - Explicitly adds common prefixes (like `ال`, `و`, `ف`, etc.) and suffixes (like `ه`, `ها`, `هم`, etc.) with the segmentation marker to the vocabulary.

4. **Enhanced Logging and Statistics**:
   - Added logging for vocabulary statistics (total unique base words, unique full words, and the final vocabulary size) to monitor the impact of the threshold and capping.
   - These logs provide insights that will help adjust parameters when scaling up to larger datasets (e.g., on your HPC).

5. **Workflow Integration**:
   - The new analysis and capping occur before extending the tokenizer’s vocabulary.
   - The tokenizer is then updated with the final vocabulary, the embeddings are resized, and both the tokenizer and model are saved.
"""

import os
import re
import gc
import json
import psutil
import shutil
import logging
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tokenizers import AddedToken
from functools import reduce

logging.basicConfig(
    filename="Tokenization.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

logging.info("Logging initialized successfully!")

MODEL_NAME = "answerdotai/ModernBERT-base"

BASE_DIR = "./Training2/"
TOKENIZER_PATH = os.path.join(BASE_DIR, "Tokenizer")
MODEL_PATH = os.path.join(BASE_DIR, "Model")
SPLITS = {
    "train": os.path.join(BASE_DIR, "train"),

}
OUTPUT_ROOT = os.path.join(BASE_DIR, "tokenized")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def memory_usage():
   """
    Get the current process memory usage.
    
    Returns:
        str: Memory usage in MB formatted as a string.
    """
    return f"{psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB"

def text_generator(input_dirs):
   """
    Generator function that yields lines of text from all .txt files in the specified directories.
    
    Args:
        input_dirs (dict): Dictionary mapping split names to directory paths.
    
    Yields:
        str: Each stripped line of text from the files.
    """
    for split, input_dir in input_dirs.items():
        if not os.path.exists(input_dir):
            continue

        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            if file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        yield line.strip()

def remove_special_tokens(text):
   """
    Remove special segmentation tokens (prefixes and suffixes) from Arabic text.
    
    This function uses regex patterns to remove affixes with segmentation markers.
    
    Args:
        text (str): Input text to clean.
    
    Returns:
        str: The text with special tokens removed.
    """
    prefixes = ['ال', 'و', 'ف', 'ب', 'ل', 'ك', 'س']
    prefix_pattern = re.compile(r'\b(?:'+'|'.join([x + '\+' for x in prefixes])+r')')
    suffixes = ['ه', 'ها', 'هم', 'نا', 'كم', 'تم', 'ون', 'ين', 'ات', 'ة', 'وا']
    suffixes_pattern = re.compile(r'(?:'+'|'.join(['\+' + x for x in suffixes])+r')\b')
    return re.sub(suffixes_pattern, '', re.sub(prefix_pattern, '', text))

def analyze_vocab_distribution(text_generator, min_freq=20, max_vocab_size=80000):
   """
    Analyze and filter vocabulary distribution from a text generator.
    
    This function computes the frequency distribution of both base words and full words 
    using a refined Arabic regex pattern. It then collects common words based on a 
    minimum frequency threshold and a high-frequency cutoff. Special prefix and suffix 
    tokens (with segmentation markers) are explicitly added. Finally, the vocabulary 
    is capped to a maximum size.
    
    Args:
        text_generator (callable): A generator function yielding text strings.
        min_freq (int, optional): Minimum frequency a base word must have to be included.
                                  Defaults to 20.
        max_vocab_size (int, optional): Maximum size of the vocabulary. Defaults to 80000.
    
    Returns:
        tuple:
            - Counter: Frequency count of base words.
            - Counter: Frequency count of full words.
            - dict: Vocabulary statistics including total unique base words, unique full words,
                    and the final vocabulary size.
            - set: The final set of common words (the vocabulary).
    """
    arabic_pattern = re.compile(r'[\u0621-\u063A\u0641-\u064A]+')  

    base_word_freq = Counter()
    full_word_freq = Counter()

    for text in text_generator(SPLITS):
        full_words = re.findall(r'\S+', text)
        full_word_freq.update(w for w in full_words if arabic_pattern.match(remove_special_tokens(w)))

        base_words = re.findall(arabic_pattern, remove_special_tokens(text))
        base_word_freq.update(base_words)

    common_words = set()

    common_words.update(word for word, freq in base_word_freq.items() if freq >= min_freq)

    common_words.update(word for word, freq in full_word_freq.most_common(10000))

    prefixes = ['ال', 'و', 'ف', 'ب', 'ل', 'ك', 'س']
    suffixes = ['ه', 'ها', 'هم', 'نا', 'كم', 'تم', 'ون', 'ين', 'ات', 'وا']

    for prefix in prefixes:
        common_words.add(f"{prefix}+")
    for suffix in suffixes:
        common_words.add(f"+{suffix}")

    if len(common_words) > max_vocab_size:
        common_words = set(list(common_words)[:max_vocab_size])  

    vocab_stats = {
        'total_unique_base_words': len(base_word_freq),
        'total_unique_full_words': len(full_word_freq),
        'final_vocab_size': len(common_words)
    }

    return base_word_freq, full_word_freq, vocab_stats, common_words

logging.info("Analyzing vocabulary distribution...")
base_freq, full_freq, stats, common_words = analyze_vocab_distribution(text_generator)

logging.info(f"Vocabulary statistics: {stats}")

logging.info("Loading tokenizer for vocabulary augmentation...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

logging.info("Adding special tokens for handling segmentation...")
special_tokens = {
    'additional_special_tokens': ['+']
}
tokenizer.add_special_tokens(special_tokens)

shutil.rmtree(TOKENIZER_PATH, ignore_errors=True)
num_added_tokens = tokenizer.add_tokens(list(common_words))
logging.info(f"Added {num_added_tokens} new Arabic tokens.")

logging.info("Saving tokenizer...")
tokenizer.save_pretrained(TOKENIZER_PATH, legacy_format=False)

logging.info("Resizing model embeddings with updated tokenizer...")
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

logging.info("Saving model...")
model.save_pretrained(MODEL_PATH)

logging.shutdown()
