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

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import yaml

# Add src/ to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from src.pretraining.trainer import (
    log_memory_usage,
    setup_performance_optimizations,
    train_model,
)
from src.utils.logging import setup_logging
from transformers import AutoTokenizer


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
        default=str(REPO_ROOT / 'models' / 'tokenizer_extension' / 'Model'),
        help='Path to model with extended tokenizer'
    )
    
    parser.add_argument(
        '--train-dir',
        type=str,
        default=str(REPO_ROOT / 'data' / 'preprocessed' / 'splits' / 'train'),
        help='Directory containing training text files'
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        default=str(REPO_ROOT / 'data' / 'preprocessed' / 'splits' / 'validation'),
        help='Directory containing validation text files'
    )

    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default=str(REPO_ROOT / 'models' / 'tokenizer_extension' / 'Tokenizer'),
        help='Path to tokenizer to load'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(REPO_ROOT / 'models' / 'output'),
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
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    
    parser.add_argument(
        '--grad-acc-steps',
        type=int,
        default=2,
        help='Gradient accumulation steps'
    )
    
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.001,
        help='Warmup ratio (fraction of total steps)'
    )
    
    # Checkpointing
    parser.add_argument(
        '--last-step',
        type=int,
        default=0,
        help='Last completed step for resuming'
    )
    
    parser.add_argument(
        '--save-checkpoint-steps',
        type=int,
        default=10000,
        help='Save checkpoint every N steps'
    )
    
    # Other options
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Disable mixed precision training'
    )
    
    parser.add_argument(
        '--no-torch-compile',
        action='store_true',
        help='Disable torch.compile for optimization'
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
        default=str(REPO_ROOT / 'logs' / 'pretraining' / 'training'),
        help='Directory to save log files'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for pretraining script."""
    args = parse_args()
    
    # Setup logging directory and descriptive filename
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_path).name
    train_name = Path(args.train_dir).name
    val_name = Path(args.val_dir).name
    log_filename = (
        f"pretraining_{model_name}_{train_name}-{val_name}_"
        f"{args.num_epochs}ep_bs{args.batch_size}_lr{args.learning_rate}_ml{args.max_length}_{timestamp}.log"
    )
    log_file = str(log_dir / log_filename)
    
    setup_logging(level=getattr(logging, args.log_level), log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    
    # Performance and environment setup (mirrors trainer helpers)
    setup_performance_optimizations(logger)
    
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
    
    if not args.train_dir or not args.val_dir:
        logger.error("--train-dir and --val-dir are required (or specify in config file)")
        sys.exit(1)
    if not args.tokenizer_path:
        logger.error("--tokenizer-path is required (or specify in config file)")
        sys.exit(1)
    
    # Validate paths
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    if not train_dir.exists():
        logger.error(f"Training directory not found: {train_dir}")
        sys.exit(1)
    if not val_dir.exists():
        logger.error(f"Validation directory not found: {val_dir}")
        sys.exit(1)
    
    # Check for text files
    if not list(train_dir.glob('*.txt')):
        logger.error(f"No .txt files found in {train_dir}")
        sys.exit(1)
    if not list(val_dir.glob('*.txt')):
        logger.error(f"No .txt files found in {val_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log device information
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA devices")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"CUDA Device {i}: {props.name}, Compute: {props.major}.{props.minor}, "
                f"Memory: {props.total_memory/1e9:.2f} GB"
            )
    
    # Display configuration
    logger.info("=" * 80)
    logger.info("ModernAraBERT Pretraining")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Train dir: {train_dir}")
    logger.info(f"Val dir: {val_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("-" * 80)
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max sequence length: {args.max_length}")
    logger.info(f"Gradient accumulation: {args.grad_acc_steps}")
    logger.info(f"Warmup ratio: {args.warmup_ratio}")
    logger.info(f"FP16: {not args.no_fp16}")
    logger.info(f"Torch compile: {not args.no_torch_compile}")
    logger.info(f"Last step: {args.last_step}")
    
    logger.info("=" * 80)
    
    # Run pretraining
    try:
        # Pre-training memory snapshot
        logger.info("Pre-training memory usage snapshot:")
        log_memory_usage(logger)
        
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        train_model(
            tokenizer=tokenizer,
            train_dir=str(train_dir),
            val_dir=str(val_dir),
            model_path=str(model_path),
            output_dir=str(output_dir),
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            grad_acc_steps=args.grad_acc_steps,
            num_workers=max(1, os.cpu_count() // 4) if os.cpu_count() else 4,
            save_checkpoint_steps=args.save_checkpoint_steps,
            warmup_ratio=args.warmup_ratio,
            last_step=args.last_step,
            fp16=not args.no_fp16,
            logging_steps=100,
            use_torch_compile=not args.no_torch_compile,
            seed=42,
        )
        
        # Post-training memory snapshot
        logger.info("Post-training memory usage snapshot:")
        log_memory_usage(logger)
        
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

