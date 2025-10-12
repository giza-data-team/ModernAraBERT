#!/usr/bin/env python3
"""
User-Facing Script: ModernAraBERT Pretraining

This script runs the full ModernAraBERT pretraining with:
- Masked Language Modeling (MLM) objective
- Multi-stage training (128 then 512 sequence length)
- Distributed training with Accelerate
- Mixed precision (FP16)
- Checkpointing and resuming

Usage Examples:
    # Train with config file (recommended)
    python scripts/pretraining/run_pretraining.py \\
        --config configs/pretraining_config.yaml
    
    # Train with command-line arguments
    python scripts/pretraining/run_pretraining.py \\
        --model-path models/modernarabert_extended \\
        --data-dir data/processed/splits/train \\
        --output-dir outputs/pretraining \\
        --num-epochs 3 \\
        --batch-size 32
    
    # Resume from checkpoint
    python scripts/pretraining/run_pretraining.py \\
        --config configs/pretraining_config.yaml \\
        --resume-from outputs/pretraining/checkpoint-1000
    
    # Multi-GPU training
    accelerate launch scripts/pretraining/run_pretraining.py \\
        --config configs/pretraining_config.yaml
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import yaml

# Add src/ to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pretraining.trainer import (
    train_mlm,
    setup_logging
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ModernAraBERT pretraining with MLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file (overrides other arguments)'
    )
    
    # Model and data
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to model with extended tokenizer'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing training text files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(REPO_ROOT / 'outputs' / 'pretraining'),
        help='Directory to save checkpoints and logs'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size per device'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-5,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=128,
        help='Maximum sequence length'
    )
    
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Gradient accumulation steps'
    )
    
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=5000,
        help='Number of warmup steps'
    )
    
    # Checkpointing
    parser.add_argument(
        '--resume-from',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--save-steps',
        type=int,
        default=1000,
        help='Save checkpoint every N steps'
    )
    
    # Other options
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='Use mixed precision training (FP16)'
    )
    
    parser.add_argument(
        '--torch-compile',
        action='store_true',
        help='Use torch.compile for optimization'
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
    """Main entry point for pretraining script."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    # Load config if provided
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key.replace('-', '_')):
                setattr(args, key.replace('-', '_'), value)
    
    # Validate required arguments
    if not args.model_path:
        logger.error("--model-path is required (or specify in config file)")
        sys.exit(1)
    
    if not args.data_dir:
        logger.error("--data-dir is required (or specify in config file)")
        sys.exit(1)
    
    # Validate paths
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Check for text files
    text_files = list(data_dir.glob('*.txt'))
    if not text_files:
        logger.error(f"No .txt files found in {data_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display configuration
    logger.info("=" * 80)
    logger.info("ModernAraBERT Pretraining")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Training files: {len(text_files)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("-" * 80)
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Warmup steps: {args.warmup_steps}")
    logger.info(f"FP16: {args.fp16}")
    logger.info(f"Torch compile: {args.torch_compile}")
    
    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from}")
    
    logger.info("=" * 80)
    
    # Run pretraining
    try:
        train_mlm(
            model_path=str(model_path),
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            save_steps=args.save_steps,
            fp16=args.fp16,
            use_compile=args.torch_compile,
            resume_from_checkpoint=args.resume_from
        )
        
        logger.info("=" * 80)
        logger.info("✅ Pretraining completed successfully!")
        logger.info(f"Final model saved to: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Pretraining failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

