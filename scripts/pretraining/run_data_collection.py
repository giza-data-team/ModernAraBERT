#!/usr/bin/env python3
"""
User-Facing Script: Data Collection for ModernAraBERT Pretraining

This script provides a simple interface to download pretraining datasets
from the links specified in data/links.json.

Usage Examples:
    # Download all datasets to default location
    python scripts/pretraining/run_data_collection.py
    
    # Download to custom directory
    python scripts/pretraining/run_data_collection.py --output-dir /path/to/data
    
    # Use custom links file
    python scripts/pretraining/run_data_collection.py --links-json custom_links.json
"""

import sys
from pathlib import Path
import argparse
import logging

# Add src/ to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pretraining.data_collection import download_and_extract_all_datasets
from src.utils.logging import setup_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download ModernAraBERT pretraining datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--links-json',
        type=str,
        default=str(REPO_ROOT / 'data' / 'links.json'),
        help='Path to JSON file containing dataset download links'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(REPO_ROOT / 'data' / 'raw'),
        help='Directory to save downloaded datasets'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for data collection script."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level), log_file="./logs/data_collection.log")
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    links_file = Path(args.links_json)
    if not links_file.exists():
        logger.error(f"Links file not found: {links_file}")
        logger.info(f"Expected location: {REPO_ROOT / 'data' / 'links.json'}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display configuration
    logger.info("=" * 80)
    logger.info("ModernAraBERT Data Collection")
    logger.info("=" * 80)
    logger.info(f"Links file: {links_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    # Run data collection
    try:
        download_and_extract_all_datasets(
            links_json_path=str(links_file),
            output_dir=str(output_dir)
        )
        logger.info("=" * 80)
        logger.info("✅ Data collection completed successfully!")
        logger.info(f"Downloaded datasets are in: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

