"""
Data Collection Module for ModernAraBERT Pretraining

This module handles downloading datasets from various sources including:
- Google Drive links
- Direct HTTP/HTTPS URLs
- Wikipedia dumps
- Compressed archives (bz2, rar, zip)

Original file: "Data collection and preprocessing.py"
Status: Logic unchanged, only extracted download functions
"""

import os
import gdown
import requests
import bz2
import rarfile
from tqdm import tqdm
import json
import logging



def setup_logging(level=logging.INFO, log_file="data_collection.log"):
    """
    Configure logging for the data collection process.
    
    Args:
        level: Logging level (default: logging.INFO)
        log_file: Path to log file (default: data_collection.log)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info("Logging initialized for data collection")



def create_directory(directory_path):
    """
    Create a directory if it does not exist, and log the operation.

    Args:
        directory_path (str): The path of the directory to create.
    """
    logging.info(f"Attempting to create directory: {directory_path}")
    os.makedirs(directory_path, exist_ok=True)
    logging.info(f"Created directory: {directory_path}")


def download_from_drive(drive_link, output_path):
    """
    Download a file from Google Drive using its shareable link and save to the output path.

    Args:
        drive_link (str): The shareable Google Drive link.
        output_path (str): The path where the downloaded file will be saved.

    Raises:
        ValueError: If the provided drive link is invalid.
    """
    logging.info(f"Starting download_from_drive with link: {drive_link}, output_path: {output_path}")
    try:
        if '/d/' in drive_link:
            file_id = drive_link.split('/d/')[1].split('/view')[0]
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                output_path,
                quiet=False
            )
            logging.info(f"Downloaded (Drive): {output_path}")
        else:
            raise ValueError("Invalid Google Drive link.")
    except IndexError:
        logging.info(f"Error: Could not process link - {drive_link}")


def download_direct_link_binary(link, output_path):
    """
    Download a file in binary mode from a direct link and save it to output_path.

    Args:
        link (str): The direct URL to the file.
        output_path (str): The output file path where the binary data will be written.
    """
    logging.info(f"Starting download_direct_link_binary with link: {link}, output_path: {output_path}")
    response = requests.get(link, stream=True)
    with open(output_path, "wb") as file:
        for chunk in tqdm(response.iter_content(chunk_size=1024)):
            if chunk:
                file.write(chunk)
    logging.info(f"Downloaded (binary): {output_path}")


def download_direct_link_text(link, output_path):
    """
    Download a file in text mode from a direct link and save it as UTF-8 encoded text.

    Args:
        link (str): The direct URL to the text file.
        output_path (str): The file path to save the downloaded text.
    """
    logging.info(f"Starting download_direct_link_text with link: {link}, output_path: {output_path}")
    response = requests.get(link, stream=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for chunk in tqdm(response.iter_content(chunk_size=1024)):
            if chunk:
                file.write(chunk.decode('utf-8', errors='ignore'))
    logging.info(f"Downloaded (text): {output_path}")


def extract_bz2(file_path, output_path):
    """
    Extract a .bz2 compressed file to output_path.

    Args:
        file_path (str): Path to the .bz2 file.
        output_path (str): Path where the extracted content should be saved.
    """
    logging.info(f"Starting extract_bz2 for {file_path} to {output_path}")
    try:
        with bz2.BZ2File(file_path, "rb") as bz2_file:
            with open(output_path, "wb") as output_file:
                output_file.write(bz2_file.read())
        logging.info(f"Extracted (bz2): {output_path}")
    except Exception as e:
        logging.info(f"Error extracting {file_path}: {e}")


def extract_rar(file_path, output_dir):
    """
    Extract a .rar archive to output_dir.

    Args:
        file_path (str): Path to the .rar file.
        output_dir (str): Directory where the extracted content should be placed.
    """
    logging.info(f"Starting extract_rar for {file_path} to {output_dir}")    
    try:
        with rarfile.RarFile(file_path) as rar_ref:
            rar_ref.extractall(output_dir)
        logging.info(f"Extracted (rar): {file_path} to {output_dir}")
    except Exception as e:
        logging.info(f"Error extracting {file_path}: {e}")


def download_datasets_from_links(links_json_path, output_dir):
    """
    Download all datasets specified in links.json file.

    Args:
        links_json_path (str): Path to links.json configuration file
        output_dir (str): Directory to save downloaded files

    Returns:
        dict: Summary of downloaded files
    """
    logging.info(f"Loading dataset links from {links_json_path}")
    
    with open(links_json_path, 'r', encoding='utf-8') as f:
        links_data = json.load(f)
    
    create_directory(output_dir)
    
    downloaded_files = {
        'text': [],
        'xml': [],
        'failed': []
    }
    
    # Download text links
    if 'text_links' in links_data:
        logging.info(f"Processing {len(links_data['text_links'])} text links")
        for name, link in links_data['text_links'].items():
            try:
                output_path = os.path.join(output_dir, f"{name}.txt.bz2")
                
                if '/drive.google.com/' in link:
                    download_from_drive(link, output_path)
                else:
                    download_direct_link_binary(link, output_path)
                
                downloaded_files['text'].append(output_path)
            except Exception as e:
                logging.info(f"Failed to download {name}: {e}")
                downloaded_files['failed'].append((name, str(e)))
    
    # Download XML links
    if 'xml_links' in links_data:
        logging.info(f"Processing {len(links_data['xml_links'])} XML links")
        for name, link in links_data['xml_links'].items():
            try:
                output_path = os.path.join(output_dir, f"{name}.xml.bz2")
                
                if '/drive.google.com/' in link:
                    download_from_drive(link, output_path)
                else:
                    download_direct_link_binary(link, output_path)
                
                downloaded_files['xml'].append(output_path)
            except Exception as e:
                logging.info(f"Failed to download {name}: {e}")
                downloaded_files['failed'].append((name, str(e)))
    
    # Download one billion links
    if 'links_one_billion' in links_data:
        logging.info(f"Processing {len(links_data['links_one_billion'])} One Billion Word links")
        for name, link in links_data['links_one_billion'].items():
            try:
                output_path = os.path.join(output_dir, f"{name}.xml")
                
                if '/drive.google.com/' in link:
                    download_from_drive(link, output_path)
                else:
                    download_direct_link_binary(link, output_path)
                
                downloaded_files['xml'].append(output_path)
            except Exception as e:
                logging.info(f"Failed to download {name}: {e}")
                downloaded_files['failed'].append((name, str(e)))
    
    # Log summary
    logging.info(f"Download complete: {len(downloaded_files['text'])} text files, "
              f"{len(downloaded_files['xml'])} XML files, "
              f"{len(downloaded_files['failed'])} failed")
    
    return downloaded_files


def extract_all_compressed(input_dir, output_dir):
    """
    Extract all compressed files (.bz2, .rar) in input_dir to output_dir.

    Args:
        input_dir (str): Directory containing compressed files
        output_dir (str): Directory to save extracted files

    Returns:
        list: Paths to extracted files
    """
    logging.info(f"Extracting compressed files from {input_dir} to {output_dir}")
    create_directory(output_dir)
    
    extracted_files = []
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        if filename.endswith('.bz2'):
            # Extract bz2 file
            output_path = os.path.join(output_dir, filename.replace('.bz2', ''))
            extract_bz2(file_path, output_path)
            extracted_files.append(output_path)
            
        elif filename.endswith('.rar'):
            # Extract rar archive
            extract_rar(file_path, output_dir)
            # Add extracted files to list
            for extracted_file in os.listdir(output_dir):
                extracted_files.append(os.path.join(output_dir, extracted_file))
    
    logging.info(f"Extracted {len(extracted_files)} files")
    return extracted_files


def download_and_extract_all_datasets(links_json_path, output_dir):
    """
    Download all datasets from links.json and extract compressed files.
    
    This is a convenience function that combines downloading and extraction.
    
    Args:
        links_json_path (str): Path to links.json configuration file
        output_dir (str): Directory to save downloaded and extracted files
        
    Returns:
        dict: Summary of downloaded and extracted files
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting download and extraction process")
    logger.info(f"Links file: {links_json_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Download datasets
    logger.info("Phase 1: Downloading datasets...")
    download_results = download_datasets_from_links(links_json_path, output_dir)
    
    # Extract compressed files
    logger.info("Phase 2: Extracting compressed files...")
    extracted_dir = os.path.join(output_dir, "extracted")
    extracted_files = extract_all_compressed(output_dir, extracted_dir)
    
    # Combine results
    combined_results = {
        'downloaded': download_results,
        'extracted': extracted_files,
        'extracted_dir': extracted_dir
    }
    
    logger.info("Download and extraction complete!")
    logger.info(f"Downloaded: {len(download_results['text']) + len(download_results['xml'])} files")
    logger.info(f"Extracted: {len(extracted_files)} files")
    logger.info(f"Extracted files location: {extracted_dir}")
    
    return combined_results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Download pretraining datasets")
    parser.add_argument(
        "--links-json",
        type=str,
        default="./data/links.json",
        help="Path to links.json configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Directory to save downloaded files"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract compressed files after download"
    )
    
    args = parser.parse_args()
    
    # Download datasets
    logging.info("Starting dataset download...")
    results = download_datasets_from_links(args.links_json, args.output_dir)
    
    logging.info("Download Summary:")
    logging.info(f"  Text files: {len(results['text'])}")
    logging.info(f"  XML files: {len(results['xml'])}")
    logging.info(f"  Failed: {len(results['failed'])}")
    
    # Extract if requested
    if args.extract:
        logging.info("Extracting compressed files...")
        extracted_dir = os.path.join(args.output_dir, "extracted")
        extracted = extract_all_compressed(args.output_dir, extracted_dir)
        logging.info(f"Extracted {len(extracted)} files to {extracted_dir}")
    
    logging.info("Done! Check data_collection.log for details.")

