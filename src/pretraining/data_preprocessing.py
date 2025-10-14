"""
Data Preprocessing Module for ModernAraBERT Pretraining

This module handles all text preprocessing operations including:
- Arabic text normalization (diacritics, tatweel removal)
- XML text extraction
- Text file processing and filtering
- Farasa morphological segmentation
- Data splitting (train/val/test)
"""

import os
import re
import random
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
from farasa.segmenter import FarasaSegmenter
import logging

def normalize_arabic_word(word: str) -> str:
    """
    Normalize an Arabic word by removing tatweel characters.

    Args:
        word (str): The Arabic word to normalize.

    Returns:
        str: The normalized word without tatweel characters.
    """
    return re.sub(r'ـ+', '', word)


def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritical marks (tashkeel) from text.

    Args:
        text (str): Input Arabic text

    Returns:
        str: Text without diacritics
    """
    # Arabic diacritics Unicode range
    diacritics_pattern = re.compile(r'[\u064B-\u065F\u0670]')
    return diacritics_pattern.sub('', text)


def clean_text(text: str) -> str:
    """
    Clean Arabic text by removing punctuation and extra whitespace.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    return re.sub(r'\s+', ' ', re.sub(r'[.,;،؛]', '', text)).strip()


def extract_text_blocks(input_directory: str, output_directory: str):
    """
    Extract <Text> blocks from XML files in the input directory and save each valid block as a line in a text file.

    The function checks that each extracted block has between 100 and 8000 words (inclusive).
    
    Args:
        input_directory (str): Directory containing XML files.
        output_directory (str): Directory where processed text files will be saved.
    """
    logging.info(f"Starting extract_text_blocks from {input_directory} to {output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    pattern = re.compile(r"<Text>(.*?)</Text>", flags=re.DOTALL)

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(".xml"):
            xml_file_path = os.path.join(input_directory, filename)
            base_name, _ = os.path.splitext(filename)
            output_file_path = os.path.join(output_directory, f"processed_{base_name}.txt")

            logging.info(f"Processing XML file: {xml_file_path}")

            with open(xml_file_path, 'r', encoding='utf-8') as f_in:
                content = f_in.read()

            matches = pattern.findall(content)

            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                for match in matches:
                    cleaned_text = match.strip()
                    lines = cleaned_text.splitlines()

                    chunk_buffer = []
                    for line in lines:
                        # If we encounter an empty line, check the accumulated chunk
                        if not line.strip():
                            # Check word count in the chunk
                            if chunk_buffer:
                                combined_chunk = " ".join(chunk_buffer).strip()
                                normalized_chunk = normalize_arabic_word(combined_chunk)
                                word_count = len(normalized_chunk.split())
                                if word_count >= 100 or word_count <= 8000:
                                    f_out.write(normalized_chunk + "\n")
                            # Reset for the next chunk
                            chunk_buffer = []
                        else:
                            # Accumulate lines
                            chunk_buffer.append(line.strip())

                    # After finishing all lines, do one last check for the remaining buffer
                    if chunk_buffer:
                        combined_chunk = " ".join(chunk_buffer).strip()
                        normalized_chunk = normalize_arabic_word(combined_chunk)
                        word_count = len(normalized_chunk.split())
                        if word_count >= 100 or word_count <= 8000:
                            f_out.write(normalized_chunk + "\n")

            logging.info(f"Extracted text written to: {output_file_path}")


def process_text_files(input_directory: str, output_directory: str):
    """
    Processes text files by reading each line and applying the following logic:

    1. If a line starts with a numeric prefix (e.g., "6943697:9095924:"), remove the prefix
       (everything up to and including the second colon). Then check if the cleaned text
       contains any English words. If it does, skip the line.

    2. If a line does NOT start with a numeric prefix, use the line as is.

    3. In either case, only keep the line if it contains at least 100 words.
    
    Args:
        input_directory (str): Directory containing the original text files.
        output_directory (str): Directory where processed text files will be saved.
    """
    logging.info(f"Starting process_text_files from {input_directory} to {output_directory}")
    os.makedirs(output_directory, exist_ok=True)

    numeric_prefix_pattern = re.compile(r"^\d+:\d+:")  
    english_pattern = re.compile(r"\b[a-zA-Z]+\b")     

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".txt"):
            input_file_path = os.path.join(input_directory, file_name)
            output_file_path = os.path.join(output_directory, f"processed_{file_name}")

            logging.info(f"Processing text file: {input_file_path}")

            with open(input_file_path, "r", encoding="utf-8") as infile:
                lines = infile.readlines()

            processed_sentences = []
            seen_lines = set()

            for line in lines:
                original_line = line.strip()
                if original_line in seen_lines:
                    continue
                seen_lines.add(original_line)

                # Check if the line starts with a numeric prefix.
                if numeric_prefix_pattern.match(original_line):
                    # Remove the prefix (everything up to and including the second colon).
                    first_colon = original_line.find(":")
                    second_colon = original_line.find(":", first_colon + 1)
                    if second_colon != -1:
                        cleaned_line = original_line[second_colon + 1:].strip()
                    else:
                        cleaned_line = original_line
                    # Only skip the line if the cleaned text contains English words.
                    if english_pattern.search(cleaned_line):
                        continue
                    sentence = cleaned_line
                else:
                    # If no numeric prefix, use the line as is.
                    sentence = original_line

                # Normalize Arabic text to remove extra tatweel characters.
                sentence = normalize_arabic_word(sentence)

                # Apply the minimum word count filter.
                if len(sentence.split()) < 100:
                    continue

                processed_sentences.append(sentence)

            if processed_sentences:
                with open(output_file_path, "w", encoding="utf-8") as outfile:
                    for sentence in processed_sentences:
                        outfile.write(sentence + "\n")
                logging.info(f"Processed file: {file_name} -> {output_file_path}")
            else:
                logging.info(f"Processed file: {file_name} -> No data to write.")


def process_xml_file(input_file_path: str, output_file_path: str):
    """
    Process a single XML file by extracting text from <text> elements, combining non-empty lines,
    cleaning the text, and writing the result to a text file if it contains at least 100 words.

    Args:
        input_file_path (str): Path to the input XML file.
        output_file_path (str): Path where the processed text file will be saved.
    """
    logging.info(f"Starting process_xml_file: {input_file_path} -> {output_file_path}")
    # Skip empty or missing files
    try:
        if not os.path.exists(input_file_path) or os.path.getsize(input_file_path) == 0:
            logging.warning(f"Skipping XML file (empty or missing): {input_file_path}")
            return
    except Exception:
        logging.warning(f"Skipping XML file (stat failed): {input_file_path}")
        return

    try:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            context = ET.iterparse(input_file_path, events=("start", "end"))
            sentence_buffer = []
            for event, elem in context:
                if event == "end" and elem.tag.endswith("text"):
                    text_content = elem.text or ""
                    for line in text_content.splitlines():
                        line = line.strip()
                        arabic_words = re.findall(r'[ء-ي]+', line)
                        arabic_line = " ".join(arabic_words)

                        if arabic_line:
                            sentence_buffer.append(arabic_line)
                        else:
                            if sentence_buffer:
                                full_sentence = clean_text(" ".join(sentence_buffer))
                                normalized_sentence = normalize_arabic_word(full_sentence)
                                if len(normalized_sentence.split()) >= 100:
                                    output_file.write(normalized_sentence + "\n")
                                sentence_buffer = []
                    if sentence_buffer:
                        full_sentence = clean_text(" ".join(sentence_buffer))
                        normalized_sentence = normalize_arabic_word(full_sentence)
                        if len(normalized_sentence.split()) >= 100:
                            output_file.write(normalized_sentence + "\n")
                        sentence_buffer = []
                    elem.clear()
        logging.info(f"Completed process_xml_file: {input_file_path}")
    except ET.ParseError as e:
        logging.error(f"Skipping malformed XML file: {input_file_path} (ParseError: {e})")
        return


def process_xml(input_directory: str, output_directory: str):
    """
    Process all XML files in the input directory by applying process_xml_file and saving the outputs
    in the specified output directory.

    Args:
        input_directory (str): Directory containing XML files.
        output_directory (str): Directory where processed text files will be saved.
    """
    logging.info(f"Starting process_directory from {input_directory} to {output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".xml"):
            input_file_path = os.path.join(input_directory, file_name)
            base_name = os.path.splitext(file_name)[0]
            output_file_name = f"processed_{base_name}.txt"
            output_file_path = os.path.join(output_directory, output_file_name)
            try:
                process_xml_file(input_file_path, output_file_path)
            except Exception as e:
                logging.error(f"Failed processing XML file {input_file_path}: {e}")
                continue
    logging.info(f"Completed process_directory from {input_directory}")


def safe_segment(segmenter: FarasaSegmenter, text: str) -> Optional[str]:
    """
    Safely segment the input text using the provided Farasa segmenter.
    Handles UnicodeDecodeError by logging the error and returning None.

    Args:
        segmenter (FarasaSegmenter): An instance of FarasaSegmenter.
        text (str): The text to be segmented.

    Returns:
        str or None: The segmented text, or None if an error occurred.
    """
    try:
        return segmenter.segment(text)
    except UnicodeDecodeError as e:
        error_message = f"UnicodeDecodeError while segmenting text: {text[:30]}... Error: {e}"
        logging.error(error_message)
        logging.info(error_message)
        return None


def apply_segmentation_to_file(
    input_file: str,
    output_base: str,
    segmenter: FarasaSegmenter,
    batch_size: int = 100000
):
    """
    Apply Farasa segmentation to an input text file in chunks, writing each chunk to a new output file.

    Args:
        input_file (str): Path to the input text file.
        output_base (str): Base path for the output files.
        segmenter (FarasaSegmenter): The FarasaSegmenter instance for segmentation.
        batch_size (int, optional): Number of lines to process per chunk. Defaults to 100000.
    """
    logging.info(f"Starting apply_segmentation_to_file: {input_file} -> {output_base}")

    with open(input_file, "r", encoding="utf-8") as infile:
        chunk_number = 1
        while True:
            lines = []
            # Read up to batch_size lines from the file
            for _ in range(batch_size):
                line = infile.readline()
                if not line:
                    break
                stripped_line = line.strip()
                if stripped_line:
                    lines.append(stripped_line)
            # Exit loop if no more lines are found
            if not lines:
                break

            segmented_lines = []
            for line in lines:
                segmented_line = safe_segment(segmenter, line)
                if segmented_line is not None:
                    segmented_lines.append(segmented_line)

            output_file = f"{output_base}_batch_{chunk_number}.txt"
            with open(output_file, "w", encoding="utf-8") as outfile:
                outfile.write("\n".join(segmented_lines))

            chunk_msg = (
                f"\nDEBUG: --- Chunk {chunk_number} ---\n"
                f"DEBUG: Processing {len(segmented_lines)} lines\n"
                f"Segmented batch {chunk_number} with {len(segmented_lines)} lines to {output_file}\n"
            )
            logging.info(chunk_msg)
            chunk_number += 1

    logging.info(f"Completed segmentation for file: {input_file}")


def segment_data(input_directory: str, output_directory: str, batch_size: int = 1000):
    """
    Segment processed text files from input_directory into output_directory.

    Args:
        input_directory (str): Directory containing processed text files (prefixed with 'processed_').
        output_directory (str): Directory to write segmented outputs into.
        batch_size (int, optional): Number of lines to process per batch. Defaults to 1000.
    """
    logging.info(f"Starting Segmenting_data from {input_directory} to {output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    farasa_segmenter = FarasaSegmenter(interactive=True)
    file_number = 1
    for file_name in os.listdir(input_directory):
        if file_name.startswith("processed_") and file_name.endswith(".txt"):
            input_file_path = os.path.join(input_directory, file_name)
            output_base = os.path.join(output_directory, f"segmented_{file_number}")
            file_number += 1
            logging.info(f"\nProcessing segmentation for file: {file_name}")
            apply_segmentation_to_file(input_file_path, output_base, farasa_segmenter, batch_size=batch_size)
            logging.info(f"\nCompleted segmentation for file: {file_name}")
    logging.info(f"Completed Segmenting_data to directory: {output_directory}")


def split_data(
    input_directory: str,
    output_base_dir: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[str, str, str]:
    """
    Split preprocessed data files into train, validation, and test sets.

    Args:
        input_directory (str): Directory containing preprocessed text files
        output_base_dir (str): Base directory for output (train/val/test subdirs will be created)
        train_ratio (float): Ratio of training data (default: 0.9)
        val_ratio (float): Ratio of validation data (default: 0.05)
        test_ratio (float): Ratio of test data (default: 0.05)
        seed (int): Random seed for reproducibility (default: 42)

    Returns:
        Tuple[str, str, str]: Paths to train, validation, and test directories
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    random.seed(seed)
    logging.info(f"Starting data split with ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}, seed={seed}")

    # Create output directories
    train_dir = os.path.join(output_base_dir, "train")
    val_dir = os.path.join(output_base_dir, "validation")
    test_dir = os.path.join(output_base_dir, "test")

    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Collect all lines from all processed files
    all_lines = []
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_directory, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                all_lines.extend(lines)
                logging.info(f"Loaded {len(lines)} lines from {file_name}")

    # Shuffle and split
    random.shuffle(all_lines)
    total_lines = len(all_lines)

    train_end = int(total_lines * train_ratio)
    val_end = train_end + int(total_lines * val_ratio)

    train_lines = all_lines[:train_end]
    val_lines = all_lines[train_end:val_end]
    test_lines = all_lines[val_end:]

    # Write splits
    with open(os.path.join(train_dir, "train.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(train_lines))
    logging.info(f"Wrote {len(train_lines)} lines to train set")

    with open(os.path.join(val_dir, "validation.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(val_lines))
    logging.info(f"Wrote {len(val_lines)} lines to validation set")

    with open(os.path.join(test_dir, "test.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(test_lines))
    logging.info(f"Wrote {len(test_lines)} lines to test set")

    logging.info("\nData split complete:")
    logging.info(f"  Train: {len(train_lines)} lines ({len(train_lines)/total_lines*100:.1f}%)")
    logging.info(f"  Validation: {len(val_lines)} lines ({len(val_lines)/total_lines*100:.1f}%)")
    logging.info(f"  Test: {len(test_lines)} lines ({len(test_lines)/total_lines*100:.1f}%)")

    return train_dir, val_dir, test_dir
