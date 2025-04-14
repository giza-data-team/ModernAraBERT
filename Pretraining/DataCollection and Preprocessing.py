import os
import gdown
import requests
import bz2
import rarfile
import shutil
from tqdm import tqdm
import re
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import csv
import json
from farasa.segmenter import FarasaSegmenter
from datetime import datetime

def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("pipeline.log", "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")


# FILE DOWNLOADER SCRIPT

def create_directory(directory_path):
    log_event(f"Attempting to create directory: {directory_path}")
    os.makedirs(directory_path, exist_ok=True)
    log_event(f"Created directory: {directory_path}")


def download_from_drive(drive_link, output_path):
    """
    Download a file from Google Drive using its shareable link.
    """
    log_event(f"Starting download_from_drive with link: {drive_link}, output_path: {output_path}")
    try:
        if '/d/' in drive_link:
            file_id = drive_link.split('/d/')[1].split('/view')[0]
            gdown.download(f"https://drive.google.com/uc?id={file_id}",
                           output_path,
                           quiet=False)
            message = f"Downloaded (Drive): {output_path}"
            print(message)
            log_event(message)
        else:
            raise ValueError("Invalid Google Drive link.")
    except IndexError:
        error_message = f"Error: Could not process link - {drive_link}"
        print(error_message)
        log_event(error_message)


def download_direct_link_binary(link, output_path):
    """
    Download a file in binary mode (useful for compressed files).
    """
    log_event(f"Starting download_direct_link_binary with link: {link}, output_path: {output_path}")
    response = requests.get(link, stream=True)
    with open(output_path, "wb") as file:
        for chunk in tqdm(response.iter_content(chunk_size=1024)):
            if chunk:
                file.write(chunk)
    message = f"Downloaded (binary): {output_path}"
    print(message)
    log_event(message)


def download_direct_link_text(link, output_path):
    """
    Download a file in text mode, decoding each chunk to UTF-8.
    """
    log_event(f"Starting download_direct_link_text with link: {link}, output_path: {output_path}")
    response = requests.get(link, stream=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for chunk in tqdm(response.iter_content(chunk_size=1024)):
            if chunk:
                file.write(chunk.decode("utf-8", errors="replace"))
    message = f"Downloaded (as text): {output_path}"
    print(message)
    log_event(message)


def extract_rar(rar_path, extract_to):
    """
    Extract a .rar archive into the specified folder.
    """
    log_event(f"Starting extract_rar with rar_path: {rar_path}, extract_to: {extract_to}")
    with rarfile.RarFile(rar_path) as rf:
        rf.extractall(extract_to)
    message = f"Extracted RAR: {rar_path} to {extract_to}"
    print(message)
    log_event(message)


def extract_bz2(bz2_path, output_path):
    """
    Extract a .bz2 file to a specified output path (decompressed).
    """
    log_event(f"Starting extract_bz2 with bz2_path: {bz2_path}, output_path: {output_path}")
    with bz2.BZ2File(bz2_path, "rb") as bz_file:
        with open(output_path, "wb") as out_file:
            out_file.write(bz_file.read())
    message = f"Extracted BZ2: {bz2_path} to {output_path}"
    print(message)
    log_event(message)


def process_text_links(links, output_folder):
    """
    Downloads links intended to be .txt files.
    Logs the dataset name (key in the JSON) along with the process.
    """
    log_event(f"Starting process_text_links into folder: {output_folder}")
    create_directory(output_folder)

    if isinstance(links, dict):
        for key, link in links.items():
            log_event(f"Starting text download for dataset: {key}")
            text_filename = f"{key}.txt"
            output_path = os.path.join(output_folder, text_filename)

            if "drive.google.com" in link:
                download_from_drive(link, output_path)
                log_event(f"Processed text download (Drive) for dataset: {key}")
                continue

            if link.endswith(".bz2"):
                temp_bz2_path = os.path.join(output_folder, f"{key}.bz2")
                download_direct_link_binary(link, temp_bz2_path)
                extract_bz2(temp_bz2_path, output_path)
                if os.path.exists(temp_bz2_path):
                    os.remove(temp_bz2_path)
                    message = f"Removed temporary file: {temp_bz2_path}"
                    print(message)
                    log_event(message)
                log_event(f"Processed text download and extraction for dataset: {key}")
            else:
                download_direct_link_text(link, output_path)
                log_event(f"Processed text download for dataset: {key}")
    else:
        for i, link in enumerate(links, start=1):
            dataset_name = f"text_file_{i}"
            log_event(f"Starting text download for dataset: {dataset_name}")
            text_filename = f"{dataset_name}.txt"
            output_path = os.path.join(output_folder, text_filename)

            if "drive.google.com" in link:
                download_from_drive(link, output_path)
                log_event(f"Processed text download (Drive) for dataset: {dataset_name}")
                continue

            if link.endswith(".bz2"):
                temp_bz2_path = os.path.join(output_folder, f"{dataset_name}.bz2")
                download_direct_link_binary(link, temp_bz2_path)
                extract_bz2(temp_bz2_path, output_path)
                if os.path.exists(temp_bz2_path):
                    os.remove(temp_bz2_path)
                    message = f"Removed temporary file: {temp_bz2_path}"
                    print(message)
                    log_event(message)
                log_event(f"Processed text download and extraction for dataset: {dataset_name}")
            else:
                download_direct_link_text(link, output_path)
                log_event(f"Processed text download for dataset: {dataset_name}")


def process_compressed_xml_links(links, output_folder):
    """
    Downloads compressed .rar/.bz2 files, then extracts them into XML.
    Logs the dataset name (key in the JSON) along with the process.
    """
    log_event(f"Starting process_compressed_xml_links into folder: {output_folder}")
    create_directory(output_folder)
    temp_folder = os.path.join(output_folder, "temp")
    create_directory(temp_folder)

    if isinstance(links, dict):
        for key, link in links.items():
            log_event(f"Starting compressed XML download for dataset: {key}")
            if link.lower().endswith(".rar"):
                compressed_path = os.path.join(temp_folder, f"{key}.rar")
            else:
                compressed_path = os.path.join(temp_folder, f"{key}.bz2")

            if "drive.google.com" in link:
                download_from_drive(link, compressed_path)
                log_event(f"Downloaded compressed XML (Drive) for dataset: {key}")
            else:
                download_direct_link_binary(link, compressed_path)
                log_event(f"Downloaded compressed XML for dataset: {key}")

            if compressed_path.endswith(".rar"):
                extract_rar(compressed_path, output_folder)
            else:
                extracted_file_path = os.path.join(output_folder, f"{key}.xml")
                extract_bz2(compressed_path, extracted_file_path)

            if os.path.exists(compressed_path):
                os.remove(compressed_path)
                message = f"Removed temporary compressed file: {compressed_path}"
                print(message)
                log_event(message)
            log_event(f"Processed compressed XML dataset: {key}")
    else:
        for i, link in enumerate(links, start=1):
            dataset_name = f"compressed_{i}"
            log_event(f"Starting compressed XML download for dataset: {dataset_name}")
            if link.lower().endswith(".rar"):
                compressed_path = os.path.join(temp_folder, f"{dataset_name}.rar")
            else:
                compressed_path = os.path.join(temp_folder, f"{dataset_name}.bz2")

            if "drive.google.com" in link:
                download_from_drive(link, compressed_path)
                log_event(f"Downloaded compressed XML (Drive) for dataset: {dataset_name}")
            else:
                download_direct_link_binary(link, compressed_path)
                log_event(f"Downloaded compressed XML for dataset: {dataset_name}")

            if compressed_path.endswith(".rar"):
                extract_rar(compressed_path, output_folder)
            else:
                extracted_file_path = os.path.join(output_folder, f"xml_file_{i}.xml")
                extract_bz2(compressed_path, extracted_file_path)

            if os.path.exists(compressed_path):
                os.remove(compressed_path)
                message = f"Removed temporary compressed file: {compressed_path}"
                print(message)
                log_event(message)
            log_event(f"Processed compressed XML dataset: {dataset_name}")

    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder, ignore_errors=True)
        message = f"Removed temp folder: {temp_folder}"
        print(message)
        log_event(message)


# One Billion PROCESSOR SCRIPT

def remove_files(directory, prefix=None, extension=None):
    """
    Removes files in the specified directory.

    - If only `extension` is provided, removes all files with that extension.
    - If `prefix` and `extension` are provided, removes files that start with `prefix` and have the `extension`.
    - If `prefix` is provided without `extension`, removes all files that start with `prefix`.
    """
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if os.path.isfile(file_path):  
            if prefix and extension:
                if file_name.startswith(prefix) and file_name.endswith(extension):
                    os.remove(file_path)
                    log_event(f"Deleted: {file_path}")
            elif extension:
                if file_name.endswith(extension):
                    os.remove(file_path)
                    log_event(f"Deleted: {file_path}")
            elif prefix:
                if file_name.startswith(prefix):
                    os.remove(file_path)
                    log_event(f"Deleted: {file_path}")

def normalize_arabic_word(word):
    # Remove one or more occurrences of the tatweel character (ـ)
    return re.sub(r'ـ+', '', word)


def extract_text_blocks(input_directory, output_directory):
    """
    Extracts <Text> blocks from XML files in input_directory and writes them as .txt files in output_directory.
    - For each <Text> block, lines are combined until an empty line is encountered.
    - Once an empty line is found (or we're at the end), we check if the combined chunk has at least 100 words
      or up to 8000 words (the code uses >=100 or <=8000 condition).
    - If valid, write that chunk as one line to the output file.
    """
    log_event(f"Starting extract_text_blocks from {input_directory} to {output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    pattern = re.compile(r"<Text>(.*?)</Text>", flags=re.DOTALL)

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(".xml"):
            xml_file_path = os.path.join(input_directory, filename)
            base_name, _ = os.path.splitext(filename)
            output_file_path = os.path.join(output_directory, f"processed_{base_name}.txt")

            log_event(f"Processing XML file: {xml_file_path}")

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

            log_event(f"Extracted text written to: {output_file_path}")
            print(f"Extracted text written to: {output_file_path}")


# TEXT PROCESSOR SCRIPT

def process_text_files(input_directory, output_directory):
    """
    Processes text files by reading each line and applying the following logic:

    1. If a line starts with a numeric prefix (e.g., "6943697:9095924:"), remove the prefix
       (everything up to and including the second colon). Then check if the cleaned text
       contains any English words. If it does, skip the line.

    2. If a line does NOT start with a numeric prefix, use the line as is.

    3. In either case, only keep the line if it contains at least 100 words.
    """
    log_event(f"Starting process_text_files from {input_directory} to {output_directory}")
    os.makedirs(output_directory, exist_ok=True)

    numeric_prefix_pattern = re.compile(r"^\d+:\d+:")  
    english_pattern = re.compile(r"\b[a-zA-Z]+\b")     

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".txt"):
            input_file_path = os.path.join(input_directory, file_name)
            output_file_path = os.path.join(output_directory, f"processed_{file_name}")

            log_event(f"Processing text file: {input_file_path}")

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
                message = f"Processed file: {file_name} -> {output_file_path}"
                print(message)
                log_event(message)
            else:
                message = f"Processed file: {file_name} -> No data to write."
                print(message)
                log_event(message)


# XML PROCESSOR SCRIPT

def clean_text(text):
    """
    Cleans Arabic text by removing specified punctuation and extra whitespace.
    """
    return re.sub(r'\s+', ' ', re.sub(r'[.,;،؛]', '', text)).strip()


def process_xml_file(input_file_path, output_file_path):
    """
    Parses an XML file, extracts Arabic words from each text element,
    combines consecutive non-empty lines into a full sentence (stopping at an empty line),
    cleans the sentence using clean_text, and writes the sentence to the output file
    only if it contains at least 100 words.
    """
    log_event(f"Starting process_xml_file: {input_file_path} -> {output_file_path}")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        context = ET.iterparse(input_file_path, events=("start", "end"))
        sentence_buffer = []
        for event, elem in context:
            if event == "end" and elem.tag.endswith("text"):
                text_content = elem.text or ""
                for line in text_content.splitlines():
                    line = line.strip()
                    # Extract Arabic words from the line
                    arabic_words = re.findall(r'[ء-ي]+', line)
                    arabic_line = " ".join(arabic_words)

                    if arabic_line:
                        sentence_buffer.append(arabic_line)
                    else:
                        if sentence_buffer:
                            full_sentence = clean_text(" ".join(sentence_buffer))
                            # Normalize the sentence to remove extra tatweel characters
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
    log_event(f"Completed process_xml_file: {input_file_path}")


def process_xml(input_directory, output_directory):
    """
    Processes all XML files in the input_directory, applies the extraction and cleaning process,
    and writes the output for each file to the output_directory with a .txt extension.
    """
    log_event(f"Starting process_directory from {input_directory} to {output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".xml"):
            input_file_path = os.path.join(input_directory, file_name)
            base_name = os.path.splitext(file_name)[0]
            output_file_name = f"processed_{base_name}.txt"
            output_file_path = os.path.join(output_directory, output_file_name)
            process_xml_file(input_file_path, output_file_path)
    log_event(f"Completed process_directory from {input_directory}")


# SEGMENTATION PROCESSOR SCRIPT

def safe_segment(segmenter, text):
    """
    Attempts to segment text using Farasa.
    """
    try:
        return segmenter.segment(text)
    except UnicodeDecodeError as e:
        error_message = f"UnicodeDecodeError while segmenting text: {text[:30]}... Error: {e}"
        print(error_message)
        log_event(error_message)
        return None


def apply_segmentation_to_file(input_file, output_base, segmenter, batch_size=100000):
    """
    Reads a text file in chunks of batch_size lines, applies Farasa segmentation to each line,
    and writes each processed chunk to a separate output file.

    The function reads N rows from the file, processes them, writes the output,
    then continues with the next N rows until the file is fully processed.
    """
    log_event(f"Starting apply_segmentation_to_file: {input_file} -> {output_base}")

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
            print(chunk_msg)
            log_event(chunk_msg)
            chunk_number += 1

    log_event(f"Completed segmentation for file: {input_file}")


def Segmenting_data(output_directory, batch_size=1000):
    """
    Processes all text files in the output_directory whose names match specific patterns.
    For each file, segmentation is applied in chunks of batch_size lines, and segmented output
    is written to separate files.
    """
    log_event(f"Starting Segmenting_data in directory: {output_directory}")
    farasa_segmenter = FarasaSegmenter(interactive=True)
    file_number = 1
    for file_name in os.listdir(output_directory):
        if file_name.startswith("processed_") and file_name.endswith(".txt"):
            input_file_path = os.path.join(output_directory, file_name)
            output_base = os.path.join(output_directory, f"segmented_{file_number}")
            file_number += 1
            debug_msg = f"\nDEBUG: Processing segmentation for file: {file_name}"
            print(debug_msg)
            log_event(debug_msg)
            apply_segmentation_to_file(input_file_path, output_base, farasa_segmenter, batch_size=batch_size)
            completion_msg = f"DEBUG: Completed segmentation for file: {file_name}"
            print(completion_msg)
            log_event(completion_msg)
    log_event(f"Completed Segmenting_data in directory: {output_directory}")


# MAIN SCRIPT PIPELINE

def main():
    log_event("Starting main script execution.")
    text_dir = "txt_files"
    xml_dir = "xml_files"

    with open("links.json", "r") as f:
        links = json.load(f)

    base_directory = Path.cwd()
    txt_input_directory = base_directory / "txt_files"
    xml_input_directory = base_directory / "xml_files"
    output_directory = base_directory / "output"
    extract_directory = base_directory / "extracted"

    if not extract_directory.exists():
      # Process One Billion datasets (RAR files)
      for key, link in links["links_one_billion"].items():
          log_event(f"Starting processing for One Billion dataset: {key}")
          rar_filename = f"{key}.rar"
          message = f"Downloading {link} as {rar_filename} for dataset: {key}"
          print(message)
          log_event(message)
          download_from_drive(link, rar_filename)
          log_event(f"Downloaded dataset: {key} from {link}")
          extract_folder = "extracted"
          extract_rar(rar_filename, extract_folder)
          log_event(f"Extracted dataset: {key} to folder: {extract_folder}")
          print("Extraction complete!")
          remove_files(base_directory, "one_billion_", ".rar")
          print("Removed .rar files\n")
          log_event(f"Removed .rar files from {base_directory}")
    else:
      log_event("Skipping One Billion XML dataset downloading files and extraction.")


    if not xml_input_directory.exists():
      # Process XML compressed files
      print("Downloading XML compressed files...")
      log_event("Starting download of XML compressed files")
      process_compressed_xml_links(links["xml_links"], xml_dir)
    else:
      log_event("Skipping XML compressed files download.")

    if not txt_input_directory.exists():
      # Process text links
      print("Downloading text files into text directory...")
      log_event("Starting download of text files")
      process_text_links(links["text_links"], text_dir)
      print("All files Downloaded!\n")
      log_event("Completed downloading all text files")
    else:
      log_event("Skipping Text files download.")

    if not output_directory.exists():
      log_event("Starting extraction of text blocks from One Billion XML files.")
      extract_text_blocks(extract_directory, output_directory)
      print("One Billion XML files are extracted.\n")
      log_event("Extracted text blocks from XML files")
    else:
      log_event("Skipping One Billion Processing.")

    process_text_files(txt_input_directory, output_directory)
    print("Text files are processed.\n")
    log_event("Processed and balanced text files")

    if not output_directory.exists():
      process_xml(xml_input_directory, output_directory)
      print("XML files are processed.\n")
      log_event("Processed XML files and generated label processed outputs")
    else:
      log_event("Skipping Xml Processing.")

    Segmenting_data(output_directory)
    print("All files are segmented.\n")
    log_event("Segmented data files")
    log_event("Main script execution completed.")


if __name__ == "__main__":
    main()