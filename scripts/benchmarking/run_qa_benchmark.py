#!/usr/bin/env python3
"""
QA Benchmarking Script - User-friendly interface for question answering benchmarking

This script provides a streamlined interface for:
- Model training and evaluation on Arabic QA datasets (Arabic-SQuAD + ARCD)
- Results analysis and metrics reporting

Usage:
    python scripts/benchmarking/run_qa_benchmark.py --model-name modernarabert
    python scripts/benchmarking/run_qa_benchmark.py --model-name arabert --epochs 5 --batch-size 16
    python scripts/benchmarking/run_qa_benchmark.py --model-name mbert --eval-metric em
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from benchmarking.qa.qa_benchmark import run_qa_benchmark
from utils.logging import setup_logging


# Colors for output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_colored(message: str, color: str = Colors.END):
    """Print colored message"""
    print(f"{color}{message}{Colors.END}")


def main():
    parser = argparse.ArgumentParser(
        description="QA Benchmarking Script - Streamlined interface for question answering benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run QA benchmark with ModernAraBERT
  python scripts/benchmarking/run_qa_benchmark.py --model-name modernarabert
  
  # Run with different model and parameters
  python scripts/benchmarking/run_qa_benchmark.py --model-name arabert --epochs 5 --batch-size 16
  
  # Use different evaluation metric
  python scripts/benchmarking/run_qa_benchmark.py --model-name mbert --eval-metric em
  
  # Run with custom model path
  python scripts/benchmarking/run_qa_benchmark.py --model-name modernarabert --model-path custom/model/path
        """
    )

    # Model configuration
    parser.add_argument("--model-name", default="modernarabert",
                        choices=["modernarabert", "arabert", "mbert", "arabert2", "marbert"],
                        help="Model to benchmark (default: modernarabert)")
    parser.add_argument("--model-path", default="gizadatateam/ModernAraBERT",
                        help="Path to pretrained model (default: gizadatateam/ModernAraBERT)")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Path to tokenizer (default: same as model)")

    # Data configuration
    parser.add_argument("--data-dir", default="./data/benchmarking/qa",
                        help="Data directory (default: ./data/benchmarking/qa)")
    parser.add_argument("--log-dir", default="./logs",
                        help="Directory for log files (default: ./logs)")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs (default: 3)")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                        help="Learning rate (default: 3e-5)")
    parser.add_argument("--encoder-lr", type=float, default=1e-5,
                        help="Learning rate for encoder layers (default: 1e-5)")
    parser.add_argument("--classifier-lr", type=float, default=5e-5,
                        help="Learning rate for classifier head (default: 5e-5)")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience (default: 2)")
    parser.add_argument("--eval-metric", type=str, default="f1", choices=["f1", "em", "sm"],
                        help="Metric for early stopping (default: f1)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Checkpointing
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save/load model checkpoint")
    parser.add_argument("--continue-from-checkpoint", action="store_true",
                        help="Load from checkpoint and continue training")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Save checkpoint every N epochs")

    # Other
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for private models")

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Setup descriptive file logging using shared utility
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"qa_benchmark_{args.model_name}_{args.epochs}ep_{timestamp}.log"
    log_file = str(log_dir / log_filename)
    setup_logging(level=logging.INFO, log_file=log_file)
    logging.info(f"Logging to: {log_file}")

    print_colored(f"{Colors.BOLD}QA Benchmarking Script{Colors.END}")
    print_colored("=" * 50, Colors.BLUE)
    print_colored(f"Model: {args.model_name} ({args.model_path})", Colors.BLUE)
    print_colored(f"Data directory: {data_dir}", Colors.BLUE)
    print_colored(f"Epochs: {args.epochs}, Batch size: {args.batch_size}", Colors.BLUE)
    print_colored(f"Eval metric: {args.eval_metric}", Colors.BLUE)

    try:
        print_colored(f"\n{Colors.BOLD}Running QA benchmark...{Colors.END}")
        print_colored("=" * 50, Colors.BLUE)

        # Run benchmark
        results = run_qa_benchmark(
            model_name=args.model_name,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            data_dir=str(data_dir),
            log_dir=args.log_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            encoder_lr=args.encoder_lr,
            classifier_lr=args.classifier_lr,
            patience=args.patience,
            eval_metric=args.eval_metric,
            checkpoint_path=args.checkpoint,
            continue_from_checkpoint=args.continue_from_checkpoint,
            save_every=args.save_every,
            hf_token=args.hf_token,
            seed=args.seed,
            num_workers=args.num_workers
        )

        # Display results
        print_colored(f"\n{Colors.BOLD}Results Summary{Colors.END}")
        print_colored("=" * 50, Colors.BLUE)
        
        if results and "results" in results:
            metrics = results["results"]
            print_colored(f"Exact Match (EM): {metrics.get('exact_match', 'N/A'):.4f}", Colors.GREEN)
            print_colored(f"F1 Score: {metrics.get('f1_score', 'N/A'):.4f}", Colors.GREEN)
            print_colored(f"Sentence Match (SM): {metrics.get('sentence_match', 'N/A'):.4f}", Colors.GREEN)
            
            if "memory_usage" in results:
                memory = results["memory_usage"]
                print_colored(f"Peak RAM: {memory.get('peak_ram_gb', 'N/A'):.2f} GB", Colors.BLUE)
                if memory.get('peak_vram_gb', 0) > 0:
                    print_colored(f"Peak VRAM: {memory.get('peak_vram_gb', 'N/A'):.2f} GB", Colors.BLUE)

        print_colored("\n✅ QA benchmark completed successfully!", Colors.GREEN)

    except Exception as e:
        print_colored(f"\n❌ QA benchmark failed: {str(e)}", Colors.RED)
        logging.error(f"QA benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
