#!/usr/bin/env python3
"""
NER Benchmarking Script - User-friendly interface for named entity recognition benchmarking

This script provides a streamlined interface that mirrors the QA/SA runners:
- Parses friendly CLI arguments with helpful defaults
- Invokes the internal NER benchmarking module
- Summarizes key metrics from the generated results JSON

Usage examples:
  python scripts/benchmarking/run_ner_benchmark.py --model modernarabert
  python scripts/benchmarking/run_ner_benchmark.py --model arabert --epochs 5 --batch-size 16
  python scripts/benchmarking/run_ner_benchmark.py --dataset asas-ai/ANERCorp --output-dir ./results/ner
"""

import sys
import os
import json
import glob
import argparse
import logging
from pathlib import Path


# Add repo root and src to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))          # Needed for 'src.*' absolute imports inside modules
sys.path.insert(0, str(REPO_ROOT / "src"))  # For direct 'benchmarking.*' imports

try:
    from benchmarking.ner import ner_benchmark as ner_module
except Exception as import_error:
    raise RuntimeError(
        "Failed to import benchmarking.ner.ner_benchmark. "
        "Ensure the 'src' directory is on PYTHONPATH."
    ) from import_error


class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_colored(message: str, color: str = Colors.END) -> None:
    print(f"{color}{message}{Colors.END}")


def build_ner_module_argv(args: argparse.Namespace) -> list[str]:
    """Build argv for ner_module.main() consistent with its argparse schema."""
    argv: list[str] = [
        "ner_benchmark",
        "--model", args.model,
        "--dataset", args.dataset,
        "--output-dir", args.output_dir,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--learning-rate", str(args.learning_rate),
        "--fine-tune", args.fine_tune,
        "--log-dir", args.log_dir,
        "--gradient-accumulation", str(args.gradient_accumulation),
    ]
    if args.inference_test:
        argv.append("--inference-test")
    return argv


def find_latest_results_json(output_dir: str) -> str | None:
    """Return the latest results JSON path in output_dir matching '*_results.json'."""
    pattern = os.path.join(output_dir, "*_results.json")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def load_metrics(results_path: str) -> dict:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NER Benchmarking Script - Streamlined interface for named entity recognition benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run NER benchmark with ModernAraBERT\n"
            "  python scripts/benchmarking/run_ner_benchmark.py --model modernarabert\n\n"
            "  # Run with different model and parameters\n"
            "  python scripts/benchmarking/run_ner_benchmark.py --model arabert --epochs 5 --batch-size 16\n\n"
            "  # Use custom dataset and output directory\n"
            "  python scripts/benchmarking/run_ner_benchmark.py --dataset asas-ai/ANERCorp --output-dir ./results/ner\n"
        ),
    )

    # Model configuration (align choices with ner_module)
    parser.add_argument("--model", default="modernarabert",
                        choices=["modernarabert", "arabert", "mbert", "arabert2", "marbert", "camel"],
                        help="Model to benchmark (default: modernarabert)")

    # Dataset configuration
    parser.add_argument("--dataset", default="asas-ai/ANERCorp",
                        help="HuggingFace dataset name (default: asas-ai/ANERCorp)")

    # Output/logging configuration
    parser.add_argument("--output-dir", default="./results/ner",
                        help="Directory to save outputs (default: ./results/ner)")
    parser.add_argument("--log-dir", default="./logs/benchmarking/ner",
                        help="Directory to save logs (default: ./logs/benchmarking/ner)")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")

    # Fine-tuning strategy
    parser.add_argument("--fine-tune", default="head-only", choices=["full", "head-only"],
                        help="Fine-tune strategy (default: head-only)")

    # Other
    parser.add_argument("--inference-test", action="store_true",
                        help="Run a sample inference test after training")

    args = parser.parse_args()

    # Ensure output directories exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging to console (module also configures file logging)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    print_colored(f"{Colors.BOLD}NER Benchmarking Script{Colors.END}")
    print_colored("=" * 50, Colors.BLUE)
    print_colored(f"Model: {args.model}", Colors.BLUE)
    print_colored(f"Dataset: {args.dataset}", Colors.BLUE)
    print_colored(f"Output directory: {args.output_dir}", Colors.BLUE)

    # Build arguments for the internal module and run
    module_argv = build_ner_module_argv(args)
    print_colored(f"\n{Colors.BOLD}Running NER benchmark...{Colors.END}")
    print_colored("=" * 50, Colors.BLUE)

    old_argv = sys.argv
    try:
        sys.argv = module_argv
        exit_code = ner_module.main()
    finally:
        sys.argv = old_argv

    if exit_code != 0:
        print_colored("\n❌ NER benchmark failed", Colors.RED)
        return 1

    # Summarize results
    print_colored(f"\n{Colors.BOLD}Results Summary{Colors.END}")
    print_colored("=" * 50, Colors.BLUE)

    results_path = find_latest_results_json(args.output_dir)
    if not results_path or not os.path.isfile(results_path):
        print_colored("Results files not found in expected location.", Colors.YELLOW)
        print_colored(f"Checked: {args.output_dir}", Colors.YELLOW)
        print_colored("✅ NER benchmark completed successfully!", Colors.GREEN)
        return 0

    try:
        data = load_metrics(results_path)
        results = data.get("results", {}) if isinstance(data, dict) else {}
        # Prefer known keys if present; otherwise print all key-values
        if results:
            # Known metric names used by the module
            macro = results.get("macro_f1_score")
            weighted = results.get("weighted_f1_score")
            if macro is not None:
                print_colored(f"Macro-F1: {macro:.4f}", Colors.GREEN)
            if weighted is not None:
                print_colored(f"Weighted-F1: {weighted:.4f}", Colors.GREEN)

            # Print any additional metrics present
            for k, v in results.items():
                if k in {"macro_f1_score", "weighted_f1_score"}:
                    continue
                try:
                    print_colored(f"{k}: {float(v):.4f}", Colors.BLUE)
                except Exception:
                    print_colored(f"{k}: {v}", Colors.BLUE)

        print_colored(f"\nJSON: {results_path}", Colors.BLUE)
        print_colored("\n✅ NER benchmark completed successfully!", Colors.GREEN)
    except Exception as e:
        print_colored(f"Failed to read results JSON: {e}", Colors.YELLOW)
        print_colored("✅ NER benchmark completed successfully!", Colors.GREEN)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


