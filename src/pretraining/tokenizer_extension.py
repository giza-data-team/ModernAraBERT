"""
Tokenizer Extension Module for ModernAraBERT

This module extends ModernBERT's vocabulary with Arabic-specific tokens:
- Analyzes Arabic text corpus to extract frequent words
- Adds 80K Arabic tokens to base vocabulary
- Handles morphological segmentation markers (prefix/suffix)
- Resizes model embeddings to accommodate new tokens

Key Features:
- Refined Arabic regex pattern for accurate word extraction
- Frequency-based vocabulary filtering
- Special token handling for Farasa segmentation (+ marker)
- Memory-efficient text processing via generators
- Vocabulary size capping (80K tokens)
"""

import os
import re
import shutil
import logging
import psutil
from collections import Counter
from typing import Dict, Generator, Tuple, Set, Any
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizer

def get_memory_usage() -> str:
    """
    Get the current process memory usage.
    
    Returns:
        str: Memory usage in MB formatted as a string.
    """
    return f"{psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB"


def text_generator(input_dirs: Dict[str, str]) -> Generator[str, None, None]:
    """
    Generator function that yields lines of text from all .txt files in the specified directories.
    
    Args:
        input_dirs (dict): Dictionary mapping split names to directory paths.
        
    Yields:
        str: Each stripped line of text from the files.
    """
    for split, input_dir in input_dirs.items():
        if not os.path.exists(input_dir):
            logging.warning(f"Directory not found: {input_dir}")
            continue

        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            if file.endswith(".txt"):
                logging.info(f"Processing file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        stripped = line.strip()
                        if stripped:
                            yield stripped


def remove_special_tokens(text: str) -> str:
    """
    Remove special segmentation tokens (prefixes and suffixes) from Arabic text.
    
    This function uses regex patterns to remove affixes with segmentation markers
    that are used by Farasa morphological analyzer.
    
    Args:
        text (str): Input text to clean.
    
    Returns:
        str: The text with special tokens removed.
    """
    # Common Arabic prefixes with segmentation marker
    prefixes = ['ال', 'و', 'ف', 'ب', 'ل', 'ك', 'س']
    prefix_pattern = re.compile(r'\b(?:' + '|'.join([x + r'\+' for x in prefixes]) + r')')
    
    # Common Arabic suffixes with segmentation marker
    suffixes = ['ه', 'ها', 'هم', 'نا', 'كم', 'تم', 'ون', 'ين', 'ات', 'ة', 'وا']
    suffixes_pattern = re.compile(r'(?:' + '|'.join([r'\+' + x for x in suffixes]) + r')\b')
    
    # Remove prefixes first, then suffixes
    text = re.sub(prefix_pattern, '', text)
    text = re.sub(suffixes_pattern, '', text)
    
    return text


def analyze_vocab_distribution(
    text_gen: Generator[str, None, None],
    min_freq: int = 20,
    max_vocab_size: int = 80000
) -> Tuple[Counter, Counter, Dict[str, int], Set[str]]:
    """
    Analyze and filter vocabulary distribution from a text generator.
    
    This function computes the frequency distribution of both base words and full words 
    using a refined Arabic regex pattern. It then collects common words based on a 
    minimum frequency threshold and a high-frequency cutoff. Special prefix and suffix 
    tokens (with segmentation markers) are explicitly added. Finally, the vocabulary 
    is capped to a maximum size.
    
    Args:
        text_gen (Generator): A generator yielding text strings.
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
    logging.info(f"Starting vocabulary analysis (min_freq={min_freq}, max_vocab_size={max_vocab_size})")
    
    # Arabic Unicode range: \u0621-\u063A (ء to غ), \u0641-\u064A (ف to ي)
    arabic_pattern = re.compile(r'[\u0621-\u063A\u0641-\u064A]+')
    
    base_word_freq = Counter()
    full_word_freq = Counter()
    
    line_count = 0
    for text in text_gen:
        line_count += 1
        if line_count % 100000 == 0:
            logging.info(f"Processed {line_count} lines. Memory usage: {get_memory_usage()}")
        
        # Extract full words (with segmentation markers)
        full_words = re.findall(r'\S+', text)
        full_word_freq.update(
            w for w in full_words 
            if arabic_pattern.match(remove_special_tokens(w))
        )
        
        # Extract base words (without segmentation markers)
        base_words = re.findall(arabic_pattern, remove_special_tokens(text))
        base_word_freq.update(base_words)
    
    logging.info(f"Vocabulary analysis complete. Processed {line_count} lines.")
    logging.info(f"Total unique base words: {len(base_word_freq)}")
    logging.info(f"Total unique full words: {len(full_word_freq)}")
    
    # Build vocabulary from frequent base words
    common_words = set()
    common_words.update(
        word for word, freq in base_word_freq.items() 
        if freq >= min_freq
    )
    
    # Add top 10K most common full words
    common_words.update(
        word for word, freq in full_word_freq.most_common(10000)
    )
    
    # Add segmentation markers for common prefixes and suffixes
    prefixes = ['ال', 'و', 'ف', 'ب', 'ل', 'ك', 'س']
    suffixes = ['ه', 'ها', 'هم', 'نا', 'كم', 'تم', 'ون', 'ين', 'ات', 'وا']
    
    for prefix in prefixes:
        common_words.add(f"{prefix}+")
    for suffix in suffixes:
        common_words.add(f"+{suffix}")
    
    logging.info(f"Vocabulary before capping: {len(common_words)} tokens")
    
    # Cap vocabulary size
    if len(common_words) > max_vocab_size:
        common_words = set(list(common_words)[:max_vocab_size])
        logging.info(f"Vocabulary capped to {max_vocab_size} tokens")
    
    vocab_stats = {
        'total_unique_base_words': len(base_word_freq),
        'total_unique_full_words': len(full_word_freq),
        'final_vocab_size': len(common_words)
    }
    
    logging.info(f"Final vocabulary statistics: {vocab_stats}")
    
    return base_word_freq, full_word_freq, vocab_stats, common_words


def extend_tokenizer_vocabulary(
    model_name: str,
    vocab_tokens: Set[str],
    output_tokenizer_path: str,
    output_model_path: str
) -> Tuple[PreTrainedTokenizer, int]:
    """
    Extend tokenizer vocabulary with new Arabic tokens and resize model embeddings.
    
    Args:
        model_name (str): Hugging Face model identifier (e.g., "answerdotai/ModernBERT-base")
        vocab_tokens (set): Set of new tokens to add to vocabulary
        output_tokenizer_path (str): Path to save extended tokenizer
        output_model_path (str): Path to save model with resized embeddings
    
    Returns:
        tuple:
            - PreTrainedTokenizer: The extended tokenizer
            - int: Number of tokens added
    """
    logging.info(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    original_vocab_size = len(tokenizer)
    logging.info(f"Original vocabulary size: {original_vocab_size}")
    
    # Add segmentation special token
    logging.info("Adding special tokens for handling segmentation...")
    special_tokens = {
        'additional_special_tokens': ['+']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Add new Arabic vocabulary
    logging.info(f"Adding {len(vocab_tokens)} new Arabic tokens to vocabulary...")
    num_added_tokens = tokenizer.add_tokens(list(vocab_tokens))
    logging.info(f"Successfully added {num_added_tokens} new tokens")
    
    new_vocab_size = len(tokenizer)
    logging.info(f"New vocabulary size: {new_vocab_size} (added {new_vocab_size - original_vocab_size} total)")
    
    # Save tokenizer
    logging.info(f"Saving extended tokenizer to: {output_tokenizer_path}")
    shutil.rmtree(output_tokenizer_path, ignore_errors=True)
    os.makedirs(output_tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(output_tokenizer_path, legacy_format=False)
    
    # Load and resize model embeddings
    logging.info(f"Loading model from: {model_name}")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    logging.info(f"Original model embedding size: {model.get_input_embeddings().weight.shape[0]}")
    
    logging.info("Resizing model embeddings to match new vocabulary...")
    model.resize_token_embeddings(len(tokenizer))
    logging.info(f"New model embedding size: {model.get_input_embeddings().weight.shape[0]}")
    
    # Save model
    logging.info(f"Saving model with resized embeddings to: {output_model_path}")
    shutil.rmtree(output_model_path, ignore_errors=True)
    os.makedirs(output_model_path, exist_ok=True)
    model.save_pretrained(output_model_path)
    
    logging.info(f"Tokenizer extension complete. Memory usage: {get_memory_usage()}")
    
    return tokenizer, num_added_tokens


def run_tokenizer_extension(
    model_name: str,
    input_dirs: Dict[str, str],
    output_base_dir: str,
    min_freq: int = 20,
    max_vocab_size: int = 80000,
    log_file: str = "tokenization.log"
) -> Dict[str, Any]:
    """
    Main pipeline for extending tokenizer vocabulary with Arabic tokens.
    
    Args:
        model_name (str): Hugging Face model identifier
        input_dirs (dict): Dictionary mapping split names to data directories
        output_base_dir (str): Base directory for saving tokenizer and model
        min_freq (int): Minimum word frequency threshold (default: 20)
        max_vocab_size (int): Maximum vocabulary size (default: 80000)
        log_file (str): Path to log file (default: tokenization.log)
    
    Returns:
        dict: Dictionary containing:
            - vocab_stats: Vocabulary statistics
            - num_added_tokens: Number of tokens added
            - tokenizer_path: Path to saved tokenizer
            - model_path: Path to saved model
    """
    # Setup
    logging.info("="*80)
    logging.info("Starting tokenizer extension pipeline")
    logging.info(f"Model: {model_name}")
    logging.info(f"Input directories: {input_dirs}")
    logging.info(f"Output directory: {output_base_dir}")
    logging.info("="*80)
    
    # Paths
    tokenizer_path = os.path.join(output_base_dir, "Tokenizer")
    model_path = os.path.join(output_base_dir, "Model")
    
    # Step 1: Analyze vocabulary
    logging.info("Step 1: Analyzing vocabulary distribution...")
    text_gen = text_generator(input_dirs)
    base_freq, full_freq, vocab_stats, common_words = analyze_vocab_distribution(
        text_gen,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size
    )
    
    # Step 2: Extend tokenizer
    logging.info("Step 2: Extending tokenizer vocabulary...")
    tokenizer, num_added_tokens = extend_tokenizer_vocabulary(
        model_name=model_name,
        vocab_tokens=common_words,
        output_tokenizer_path=tokenizer_path,
        output_model_path=model_path
    )
    
    # Summary
    logging.info("="*80)
    logging.info("Tokenizer extension pipeline completed successfully!")
    logging.info(f"Vocabulary statistics: {vocab_stats}")
    logging.info(f"Tokens added: {num_added_tokens}")
    logging.info(f"Tokenizer saved to: {tokenizer_path}")
    logging.info(f"Model saved to: {model_path}")
    logging.info("="*80)
    
    logging.shutdown()
    
    return {
        'vocab_stats': vocab_stats,
        'num_added_tokens': num_added_tokens,
        'tokenizer_path': tokenizer_path,
        'model_path': model_path
    }


def extend_tokenizer_pipeline(
    model_name: str,
    text_dir: str,
    output_dir: str,
    min_freq: int = 20,
    max_vocab_size: int = 80000,
    log_level: int = logging.INFO,
) -> Tuple[PreTrainedTokenizer, AutoModelForMaskedLM]:
    """
    Adapter providing the API expected by scripts. Runs the analysis and extension and
    returns the in-memory tokenizer and model with resized embeddings.
    """
    input_dirs = {"train": text_dir}

    text_gen = text_generator(input_dirs)
    _, _, _, common_words = analyze_vocab_distribution(
        text_gen,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )

    tokenizer, _ = extend_tokenizer_vocabulary(
        model_name=model_name,
        vocab_tokens=common_words,
        output_tokenizer_path=os.path.join(output_dir, "Tokenizer"),
        output_model_path=os.path.join(output_dir, "Model"),
    )

    # Load model that has been saved with resized embeddings
    model = AutoModelForMaskedLM.from_pretrained(os.path.join(output_dir, "Model"))
    return tokenizer, model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extend ModernBERT tokenizer with Arabic vocabulary"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Hugging Face model identifier (default: answerdotai/ModernBERT-base)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing preprocessed Arabic text files (.txt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for extended tokenizer and model"
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=20,
        help="Minimum word frequency threshold (default: 20)"
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=80000,
        help="Maximum vocabulary size (default: 80000)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="tokenization.log",
        help="Path to log file (default: tokenization.log)"
    )
    
    args = parser.parse_args()
    
    # Setup input directories (expects train/ subdirectory)
    input_dirs = {
        "train": args.input_dir if os.path.isdir(args.input_dir) else os.path.dirname(args.input_dir)
    }
    
    # Run pipeline
    results = run_tokenizer_extension(
        model_name=args.model_name,
        input_dirs=input_dirs,
        output_base_dir=args.output_dir,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        log_file=args.log_file
    )
    
    # Print summary
    print("\n" + "="*80)
    print("TOKENIZER EXTENSION COMPLETE")
    print("="*80)
    print("Vocabulary Statistics:")
    for key, value in results['vocab_stats'].items():
        print(f"  {key}: {value:,}")
    print(f"\nTokens Added: {results['num_added_tokens']:,}")
    print(f"Tokenizer Path: {results['tokenizer_path']}")
    print(f"Model Path: {results['model_path']}")
    print(f"Log File: {args.log_file}")
    print("="*80)

