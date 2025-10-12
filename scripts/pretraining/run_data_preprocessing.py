#!/usr/bin/env python3
"""
User-Facing Script: Data Preprocessing for ModernAraBERT Pretraining

This script provides a simple interface to preprocess Arabic text data
for ModernAraBERT pretraining, including:
- XML text extraction (Wikipedia dumps)
- Text file processing with Arabic normalization
- Farasa morphological segmentation
- Train/val/test data splitting

Usage Examples:
    # Full preprocessing pipeline
    python scripts/pretraining/run_data_preprocessing.py \\
        --input-dir data/raw \\
        --output-dir data/processed \\
        --all
    
    # Only process XML files (Wikipedia)
    python scripts/pretraining/run_data_preprocessing.py \\
        --input-dir data/raw/wikipedia \\
        --output-dir data/processed/wikipedia \\
        --process-xml
    
    # Process text files and segment with Farasa
    python scripts/pretraining/run_data_preprocessing.py \\
        --input-dir data/raw/osian \\
        --output-dir data/processed/osian \\
        --process-text --segment
    
    # Only split already processed data
    python scripts/pretraining/run_data_preprocessing.py \\
        --input-dir data/processed \\
        --output-dir data/final \\
        --split
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Add src/ to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pretraining.data_preprocessing import (
    extract_text_from_xml_dir,
    process_text_files_parallel,
    segment_text_files_farasa,
    split_data,
    setup_logging
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess Arabic text for ModernAraBERT pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory containing raw data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed data'
    )
    
    # Processing steps
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all preprocessing steps (XML → text → segment → split)'
    )
    
    parser.add_argument(
        '--process-xml',
        action='store_true',
        help='Extract text from XML files (Wikipedia dumps)'
    )
    
    parser.add_argument(
        '--process-text',
        action='store_true',
        help='Process text files (normalization, filtering)'
    )
    
    parser.add_argument(
        '--segment',
        action='store_true',
        help='Apply Farasa morphological segmentation'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split data into train/val/test (60/20/20)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--min-words',
        type=int,
        default=100,
        help='Minimum words per document'
    )
    
    parser.add_argument(
        '--max-words',
        type=int,
        default=8000,
        help='Maximum words per document'
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
    """Main entry point for data preprocessing script."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display configuration
    logger.info("=" * 80)
    logger.info("ModernAraBERT Data Preprocessing")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Word count range: {args.min_words}-{args.max_words}")
    logger.info("=" * 80)
    
    # Determine which steps to run
    run_xml = args.all or args.process_xml
    run_text = args.all or args.process_text
    run_segment = args.all or args.segment
    run_split = args.all or args.split
    
    if not any([run_xml, run_text, run_segment, run_split]):
        logger.error("No processing steps specified. Use --all or individual flags.")
        logger.info("Run with --help for usage information.")
        sys.exit(1)
    
    try:
        # Step 1: Extract text from XML
        if run_xml:
            logger.info("=" * 80)
            logger.info("Step 1: Extracting text from XML files")
            logger.info("=" * 80)
            xml_output = output_dir / 'extracted_text'
            extract_text_from_xml_dir(
                xml_dir=str(input_dir),
                output_dir=str(xml_output)
            )
            input_dir = xml_output  # Update input for next step
        
        # Step 2: Process text files
        if run_text:
            logger.info("=" * 80)
            logger.info("Step 2: Processing text files")
            logger.info("=" * 80)
            text_output = output_dir / 'processed_text'
            process_text_files_parallel(
                input_dir=str(input_dir),
                output_dir=str(text_output),
                min_words=args.min_words,
                max_words=args.max_words
            )
            input_dir = text_output  # Update input for next step
        
        # Step 3: Farasa segmentation
        if run_segment:
            logger.info("=" * 80)
            logger.info("Step 3: Applying Farasa segmentation")
            logger.info("=" * 80)
            segment_output = output_dir / 'segmented'
            segment_text_files_farasa(
                input_dir=str(input_dir),
                output_dir=str(segment_output)
            )
            input_dir = segment_output  # Update input for next step
        
        # Step 4: Split data
        if run_split:
            logger.info("=" * 80)
            logger.info("Step 4: Splitting data into train/val/test")
            logger.info("=" * 80)
            split_output = output_dir / 'splits'
            split_data(
                input_dir=str(input_dir),
                output_dir=str(split_output),
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2
            )
        
        logger.info("=" * 80)
        logger.info("✅ Data preprocessing completed successfully!")
        logger.info(f"Processed data is in: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Data preprocessing failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

