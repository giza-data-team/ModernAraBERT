#!/usr/bin/env python3
"""
User-Facing Script: Tokenizer Extension for ModernAraBERT

This script extends the ModernBERT tokenizer vocabulary with 80,000
Arabic-specific tokens learned from the pretraining corpus.

Usage Examples:
    # Extend tokenizer with default settings (80K tokens)
    python scripts/pretraining/run_tokenizer_extension.py \\
        --model-name answerdotai/ModernBERT-base \\
        --input-dir data/processed/segmented \\
        --output-dir models/modernarabert_tokenizer
    
    # Custom vocabulary size
    python scripts/pretraining/run_tokenizer_extension.py \\
        --model-name answerdotai/ModernBERT-base \\
        --input-dir data/processed/segmented \\
        --output-dir models/tokenizer_custom \\
        --max-vocab-size 50000
    
    # Analyze vocabulary without extending
    python scripts/pretraining/run_tokenizer_extension.py \\
        --model-name answerdotai/ModernBERT-base \\
        --input-dir data/processed/segmented
"""

import sys
from pathlib import Path
import argparse
import logging

# Add src/ to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pretraining.tokenizer_extension import extend_tokenizer_pipeline
from src.utils.logging import setup_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extend ModernBERT tokenizer with Arabic vocabulary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='answerdotai/ModernBERT-base',
        help='Base model name or path (ModernBERT)'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=False,
        default=str(REPO_ROOT / 'data' / 'preprocessed' / 'segmented'),
        help='Directory containing preprocessed Arabic text files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=False,
        default=str(REPO_ROOT / 'models' / 'modernarabert_tokenizer'),
        help='Directory to save extended tokenizer and model'
    )
    
    parser.add_argument(
        '--max-vocab-size',
        type=int,
        default=80000,
        help='Maximum number of new tokens to add (default: 80K)'
    )
    
    parser.add_argument(
        '--min-freq',
        type=int,
        default=5,
        help='Minimum frequency for a token to be included'
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
    """Main entry point for tokenizer extension script."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level), log_file="tokenizer_extension.log")
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Check for text files
    text_files = list(input_dir.glob('*.txt'))
    if not text_files:
        logger.error(f"No .txt files found in {input_dir}")
        sys.exit(1)
    
    # Display configuration
    logger.info("=" * 80)
    logger.info("ModernAraBERT Tokenizer Extension")
    logger.info("=" * 80)
    logger.info(f"Base model: {args.model_name}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Text files found: {len(text_files)}")
    logger.info(f"Max new tokens: {args.max_vocab_size:,}")
    logger.info(f"Min frequency: {args.min_freq}")
    
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Run tokenizer extension
    try:        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run extension pipeline
        extended_tokenizer, extended_model = extend_tokenizer_pipeline(
            model_name=args.model_name,
            text_dir=str(input_dir),
            output_dir=str(output_dir),
            min_freq=args.min_freq,
            max_vocab_size=args.max_vocab_size
        )
        
        logger.info("=" * 80)
        logger.info("✅ Tokenizer extension completed successfully!")
        logger.info(f"Extended tokenizer saved to: {output_dir}")
        logger.info(f"New vocabulary size: {len(extended_tokenizer):,} tokens")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("1. Use this tokenizer for pretraining with run_pretraining.py")
        logger.info("2. The model embeddings have been resized automatically")
        
    except Exception as e:
        logger.error(f"❌ Tokenizer extension failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

