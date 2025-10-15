#!/usr/bin/env python3
"""
User-Facing Script: Data Preprocessing for ModernAraBERT Pretraining

This script provides a simple interface to preprocess Arabic text data
for ModernAraBERT pretraining. It runs the complete pipeline:
- XML text extraction (Wikipedia dumps)
- Text file processing with Arabic normalization
- Farasa morphological segmentation
- Train/val/test data splitting

Usage Examples:
    # Run preprocessing pipeline with default output directory
    python scripts/pretraining/run_data_preprocessing.py \\
        --input-dir data/raw
    
    # Run preprocessing with custom output directory
    python scripts/pretraining/run_data_preprocessing.py \\
        --input-dir data/raw/wikipedia \\
        --output-dir data/processed/wikipedia
    
    # Run with debug logging
    python scripts/pretraining/run_data_preprocessing.py \\
        --input-dir data/raw \\
        --log-level DEBUG
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add src/ to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pretraining.data_preprocessing import (
    process_xml,
    process_text_files,
    segment_data,
    split_data,
)

from src.utils.logging import setup_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess Arabic text for ModernAraBERT pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./data/raw/extracted',
        help='Input directory containing extracted XML/text (default: ./data/extracted)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/preprocessed',
        help='Output directory for preprocessed data (default: ./data/preprocessed)'
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
        help='Extract text from XML files only'
    )
    parser.add_argument(
        '--process-text',
        action='store_true',
        help='Process text files only'
    )
    parser.add_argument(
        '--segment',
        action='store_true',
        help='Apply Farasa segmentation only'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split data into train/val/test only'
    )

    # Parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for segmentation'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default=str(REPO_ROOT / 'logs' / 'pretraining' / 'data_preprocessing'),
        help='Directory to save log files'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for data preprocessing script."""
    args = parse_args()
    
    # Setup logging directory and descriptive filename
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    in_name = Path(args.input_dir).name or 'input'
    steps = []
    if args.all:
        steps.append('all')
    else:
        if args.process_xml:
            steps.append('xml')
        if args.process_text:
            steps.append('text')
        if args.segment:
            steps.append('segment')
        if args.split:
            steps.append('split')
    steps_part = '-'.join(steps) if steps else 'none'
    log_filename = f"data_preprocessing_{in_name}_{steps_part}_{timestamp}.log"
    log_file = str(log_dir / log_filename)
    
    setup_logging(level=getattr(logging, args.log_level), log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    
    # Validate inputs
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine steps
    run_xml = args.all or args.process_xml
    run_text = args.all or args.process_text
    run_segment = args.all or args.segment
    run_split = args.all or args.split

    if not any([run_xml, run_text, run_segment, run_split]):
        logger.error("No processing steps specified. Use --all or individual flags.")
        logger.info("Run with --help for usage information.")
        sys.exit(1)

    # Display configuration
    logger.info("=" * 80)
    logger.info("ModernAraBERT Data Preprocessing")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    if args.all:
        logger.info("Running all preprocessing steps: XML → text → segment → split")
    else:
        enabled = []
        if run_xml:
            enabled.append('xml')
        if run_text:
            enabled.append('text')
        if run_segment:
            enabled.append('segment')
        if run_split:
            enabled.append('split')
        logger.info(f"Running selected steps: {', '.join(enabled)}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Extract text from XML
        if run_xml:
            logger.info("=" * 80)
            logger.info("Step 1: Extracting text from XML files")
            logger.info("=" * 80)
            xml_output = output_dir / 'extracted'
            xml_output.mkdir(parents=True, exist_ok=True)
            process_xml(
                input_directory=str(input_dir),
                output_directory=str(xml_output)
            )
            input_dir = xml_output  # Update input for next step
        
        # Step 2: Process text files
        if run_text:
            logger.info("=" * 80)
            logger.info("Step 2: Processing text files")
            logger.info("=" * 80)
            # If xml wasn't run this time, expect extracted as source
            if not args.all and not run_xml:
                candidate = output_dir / 'extracted'
                if not candidate.exists():
                    logger.error(f"Expected extracted dir not found: {candidate}")
                    sys.exit(1)
                input_dir = candidate
            text_output = output_dir / 'processed'
            text_output.mkdir(parents=True, exist_ok=True)
            process_text_files(
                input_directory=str(input_dir),
                output_directory=str(text_output)
            )
            input_dir = text_output  # Update input for next step
        
        # Step 3: Farasa segmentation
        if run_segment:
            logger.info("=" * 80)
            logger.info("Step 3: Applying Farasa segmentation")
            logger.info("=" * 80)
            # If text wasn't run this time, expect processed as source
            if not args.all and not run_text:
                candidate = output_dir / 'processed'
                if not candidate.exists():
                    logger.error(f"Expected processed dir not found: {candidate}")
                    sys.exit(1)
                input_dir = candidate
            segment_output = output_dir / 'segmented'
            segment_output.mkdir(parents=True, exist_ok=True)
            segment_data(
                input_directory=str(input_dir),
                output_directory=str(segment_output),
                batch_size=args.batch_size
            )
            input_dir = segment_output
        
        # Step 4: Split data
        if run_split:
            logger.info("=" * 80)
            logger.info("Step 4: Splitting data into train/val/test")
            logger.info("=" * 80)
            # If segment wasn't run this time, expect segmented as source
            if not args.all and not run_segment:
                candidate = output_dir / 'segmented'
                if not candidate.exists():
                    logger.error(f"Expected segmented dir not found: {candidate}")
                    sys.exit(1)
                input_dir = candidate
            split_output = output_dir / 'splits'
            split_output.mkdir(parents=True, exist_ok=True)
            split_data(
                input_directory=str(input_dir),
                output_base_dir=str(split_output),
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

