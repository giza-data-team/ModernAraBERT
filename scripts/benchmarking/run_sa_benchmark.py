#!/usr/bin/env python3
"""
SA Benchmarking Script - User-friendly interface for sentiment analysis benchmarking

This script provides a streamlined interface for:
- Dataset preparation and management (download + prepare)
- Model training and evaluation
- Results analysis

Usage:
    python scripts/benchmarking/run_sa_benchmark.py --datasets hard labr ajgt
    python scripts/benchmarking/run_sa_benchmark.py --stage prepare-data --datasets hard
    python scripts/benchmarking/run_sa_benchmark.py --stage benchmark --datasets hard
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from datasets import load_dataset
from benchmarking.sa.sa_benchmark import run_sa_benchmark
from benchmarking.sa.preprocessing import process_dataset
from benchmarking.sa.datasets import (
    DATASET_CONFIGS,
    prepare_labr_benchmark,
    prepare_ajgt_benchmark
)
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


def check_dataset_ready(data_dir: Path, dataset_name: str) -> bool:
    """Check if dataset is already prepared"""
    dataset_path = data_dir / dataset_name
    required_files = ["train.txt", "test.txt"]

    # Check if validation.txt is required (LABR doesn't have it)
    if dataset_name.lower() != "labr":
        required_files.append("validation.txt")

    return all((dataset_path / file).exists() for file in required_files)


def prepare_data_stage(datasets: List[str], data_dir: Path, force_redownload: bool = False) -> Dict[str, bool]:
    """Stage 1: Download and prepare datasets"""
    print_colored(f"\n{Colors.BOLD}Stage 1: Preparing datasets{Colors.END}")
    print_colored("=" * 50, Colors.BLUE)

    results = {}

    for dataset_name in datasets:
        print_colored(
            f"\nPreparing {dataset_name.upper()} dataset...", Colors.YELLOW)

        dataset_path = data_dir / dataset_name
        dataset_config = DATASET_CONFIGS.get(dataset_name.lower())

        if not dataset_config:
            print_colored(f"❌ Unknown dataset: {dataset_name}", Colors.RED)
            results[dataset_name] = False
            continue

        # Check if already prepared
        if not force_redownload and check_dataset_ready(data_dir, dataset_name):
            print_colored(
                f"✅ Dataset {dataset_name} already prepared, skipping...", Colors.GREEN)
            results[dataset_name] = True
            continue

        # Prepare dataset based on type
        try:
            if dataset_name.lower() == "hard":
                # Download from HuggingFace and process
                print_colored("  Downloading from HuggingFace...", Colors.BLUE)
                dataset = load_dataset(
                    "Elnagara/hard", "plain_text", split="train")
                print_colored(
                    "  Processing dataset (Arabic filtering, text chunking)...", Colors.BLUE)
                success = process_dataset(
                    dataset, window_size=8192, base_dir=str(dataset_path))

            elif dataset_name.lower() == "labr":
                # Download from GitHub and prepare
                print_colored("  Downloading from GitHub...", Colors.BLUE)
                success = prepare_labr_benchmark(
                    str(dataset_path), dataset_config)

            elif dataset_name.lower() == "ajgt":
                # Download XLSX from GitHub and prepare
                print_colored("  Downloading XLSX from GitHub...", Colors.BLUE)
                success = prepare_ajgt_benchmark(
                    str(dataset_path), dataset_config)

            else:
                print_colored(
                    f"❌ Unsupported dataset: {dataset_name}", Colors.RED)
                results[dataset_name] = False
                continue

            if success:
                print_colored(
                    f"✅ {dataset_name.upper()} dataset prepared successfully", Colors.GREEN)
                results[dataset_name] = True
            else:
                print_colored(
                    f"❌ Failed to prepare {dataset_name} dataset", Colors.RED)
                results[dataset_name] = False

        except Exception as e:
            print_colored(
                f"❌ Error preparing {dataset_name}: {str(e)}", Colors.RED)
            results[dataset_name] = False

    return results


def benchmark_stage(datasets: List[str], data_dir: Path, model_name: str, model_path: str,
                    tokenizer_path: Optional[str], **kwargs) -> Dict[str, bool]:
    """Stage 2: Run training and evaluation"""
    print_colored(f"\n{Colors.BOLD}Stage 2: Running benchmarks{Colors.END}")
    print_colored("=" * 50, Colors.BLUE)

    results = {}

    for dataset_name in datasets:
        print_colored(
            f"\nRunning benchmark on {dataset_name.upper()} dataset...", Colors.YELLOW)

        dataset_path = data_dir / dataset_name
        dataset_config = DATASET_CONFIGS.get(dataset_name.lower())

        if not dataset_config:
            print_colored(f"❌ Unknown dataset: {dataset_name}", Colors.RED)
            results[dataset_name] = False
            continue

        # Check if dataset is prepared
        if not check_dataset_ready(data_dir, dataset_name):
            print_colored(
                f"❌ Dataset {dataset_name} not prepared. Run prepare-data stage first.", Colors.RED)
            results[dataset_name] = False
            continue

        try:
            # Configure per-dataset file logging using shared utility
            log_dir = Path(kwargs.get("log_dir", "./logs/benchmarking/sa"))
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"sa_benchmark_{model_name}_{dataset_name}_{kwargs.get('epochs', 0)}ep_p{kwargs.get('patience', 0)}_{timestamp}.log"
            log_filepath = str(log_dir / log_filename)
            setup_logging(level=logging.INFO, log_file=log_filepath)
            logging.info(f"Logging to: {log_filepath}")

            # Run benchmark
            print_colored(
                f"  Training {model_name} on {dataset_name}...", Colors.BLUE)
            benchmark_results = run_sa_benchmark(
                model_name=model_name,
                dataset_name=dataset_name,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                data_dir=str(dataset_path),
                num_labels=dataset_config["num_labels"],
                **kwargs
            )

            print_colored(
                f"✅ {dataset_name.upper()} benchmark completed successfully", Colors.GREEN)
            print_colored(
                f"  Macro-F1: {benchmark_results['results'][model_name]['macro_f1']:.4f}", Colors.BLUE)
            results[dataset_name] = True

        except Exception as e:
            print_colored(
                f"❌ Error running benchmark on {dataset_name}: {str(e)}", Colors.RED)
            results[dataset_name] = False
        finally:
            pass

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SA Benchmarking Script - Streamlined interface for sentiment analysis benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on HARD dataset
  python scripts/benchmarking/run_sa_benchmark.py --datasets hard
  
  # Run full pipeline on multiple datasets
  python scripts/benchmarking/run_sa_benchmark.py --datasets hard labr ajgt
  
  # Run only prepare-data stage
  python scripts/benchmarking/run_sa_benchmark.py --stage prepare-data --datasets hard labr
  
  # Run only benchmark stage
  python scripts/benchmarking/run_sa_benchmark.py --stage benchmark --datasets hard
  
  # Force re-download even if files exist
  python scripts/benchmarking/run_sa_benchmark.py --datasets hard labr --force-redownload
        """
    )

    # Stage selection
    parser.add_argument("--stage", choices=["prepare-data", "benchmark"],
                        help="Run specific stage (default: run both stages)")

    # Dataset selection (required)
    parser.add_argument("--datasets", nargs="+", required=True,
                        choices=["hard", "labr", "ajgt"],
                        help="Datasets to process (space-separated)")

    # Model configuration
    parser.add_argument("--model-name", default="modernarabert",
                        choices=["modernarabert", "arabert",
                                 "mbert", "arabert2", "marbert"],
                        help="Model to benchmark (default: modernarabert)")
    parser.add_argument("--model-path", default="gizadatateam/ModernAraBERT",
                        help="Path to pretrained model (default: gizadatateam/ModernAraBERT)")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Path to tokenizer (default: same as model)")

    # Data configuration
    parser.add_argument("--data-dir", default="./data/benchmarking/sa",
                        help="Data directory (default: ./data/benchmarking/sa)")
    parser.add_argument("--output-dir", default="./results/benchmarking/sa",
                        help="Directory to save results (default: ./results/benchmarking/sa)")
    parser.add_argument("--force-redownload", action="store_true",
                        help="Re-download even if files exist (default: skip if exists)")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (default: 50)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Model options
    parser.add_argument("--no-freeze", action="store_true",
                        help="Train full model (don't freeze encoder)")

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
    parser.add_argument("--log-dir", type=str, default="./log/benchmarking/sa",
                        help="Directory for log files (default: ./log/benchmarking/sa)")

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Setup console logging; per-dataset file logging is configured inside benchmark_stage
    setup_logging(level=logging.INFO)

    # Prepare training kwargs
    training_kwargs = {
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "num_workers": args.num_workers,
        "freeze_encoder": not args.no_freeze,
        "checkpoint_path": args.checkpoint,
        "continue_from_checkpoint": args.continue_from_checkpoint,
        "save_every": args.save_every,
        "hf_token": args.hf_token,
        "log_dir": args.log_dir,
        "results_dir": args.output_dir,
        "seed": args.seed
    }

    print_colored(f"{Colors.BOLD}SA Benchmarking Script{Colors.END}")
    print_colored("=" * 50, Colors.BLUE)
    print_colored(f"Datasets: {', '.join(args.datasets)}", Colors.BLUE)
    print_colored(f"Model: {args.model_name} ({args.model_path})", Colors.BLUE)
    print_colored(f"Data directory: {data_dir}", Colors.BLUE)

    # Run stages
    if args.stage == "prepare-data":
        # Only prepare data
        results = prepare_data_stage(
            args.datasets, data_dir, args.force_redownload)

    elif args.stage == "benchmark":
        # Only run benchmarks
        results = benchmark_stage(
            args.datasets, data_dir, args.model_name, args.model_path,
            args.tokenizer_path, **training_kwargs
        )

    else:
        # Run both stages
        print_colored(f"\nRunning full pipeline...", Colors.BOLD)

        # Stage 1: Prepare data
        prepare_results = prepare_data_stage(
            args.datasets, data_dir, args.force_redownload)

        # Stage 2: Run benchmarks (only on successfully prepared datasets)
        successful_datasets = [
            d for d, success in prepare_results.items() if success]
        if successful_datasets:
            benchmark_results = benchmark_stage(
                successful_datasets, data_dir, args.model_name, args.model_path,
                args.tokenizer_path, **training_kwargs
            )
            results = {**prepare_results, **benchmark_results}
        else:
            print_colored(
                "❌ No datasets prepared successfully, skipping benchmark stage", Colors.RED)
            results = prepare_results

    # Summary
    print_colored(f"\n{Colors.BOLD}Summary{Colors.END}")
    print_colored("=" * 50, Colors.BLUE)

    for dataset, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        color = Colors.GREEN if success else Colors.RED
        print_colored(f"  {dataset.upper()}: {status}", color)

    # Exit with error if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
