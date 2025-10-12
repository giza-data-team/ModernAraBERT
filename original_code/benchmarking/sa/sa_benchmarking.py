import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import psutil
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)
import argparse
import pandas as pd
import csv
import json
from datetime import datetime
from text_preprocessing import *
from train import *

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Use environment variable for token instead of hardcoding
HF_TOKEN = "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"

parser = argparse.ArgumentParser("Sentiment Analysis Benchmarking")
parser.add_argument("--model-name", dest="model_name", type=str, default='modernbert',
                    choices=['modernbert', 'arabert', 'mbert', 'arabert2', 'marbert'])
parser.add_argument("--dataset", dest="dataset_name", type=str, default='hard',
                    choices=['hard', 'astd', 'labr', 'ajgt'])
parser.add_argument("--benchmark", dest="benchmark", action='store_true',
                    help="Use test set for evaluation instead of validation set")
parser.add_argument("--max-size", dest="max_size", type=int, default=512)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=16)
parser.add_argument("--epochs", dest="epochs", type=int, default=50)
parser.add_argument("--learning-rate", dest="learning_rate",
                    type=float, default=2e-5)
parser.add_argument("--split-ratio", dest="split_ratio", type=str, default="0.6,0.2,0.2",
                    help="Train, Test, Validation split ratios, comma separated")
parser.add_argument("--num-workers", dest="num_workers", type=int, default=2)
parser.add_argument("--freeze", dest="freeze", action='store_true', default=True,
                    help="Freeze model parameters except classifier head")
parser.add_argument("--checkpoint", dest="checkpoint", type=str, default=None,
                    help="Path to save/load model checkpoint")
parser.add_argument("--continue-from-checkpoint", dest="continue_from_checkpoint", action='store_true',
                    help="Flag to load from a saved checkpoint and continue training")
parser.add_argument("--preprocess-flag", dest="preprocess_flag", action='store_true',
                    help="If True, skip dataset preprocessing (assumes preprocessing is already done)")
parser.add_argument("--save-every", dest="save_every", type=int, default=None,
                    help="Save checkpoint every N epochs (if provided)")
parser.add_argument("--use-streaming", dest="use_streaming", action='store_true',
                    help="Use streaming mode for dataset (splitting not supported in streaming mode)")
parser.add_argument("--patience", dest="patience", type=int, default=5,
                    help="Early stopping patience value")
#parser.add_argument("--inference", dest="inference_flag", type=int, default=False,
#                    help="flag for inference on test set only")
args = parser.parse_args()

# Generate a unique log filename based on model, epochs, patience, and timestamp
def get_log_filename(model_name, epochs, patience, dataset_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #if args.inference_flag:
    #    return f"SA_inference_{model_name}_{dataset_name}_{epochs}ep_p{patience}_{timestamp}.log"
    return f"SA_Benchmark_{model_name}_{dataset_name}_{epochs}ep_p{patience}_{timestamp}.log"

# Set patience value (previously hardcoded in main function)
PATIENCE = args.patience

# Create log directory if it doesn't exist
LOG_DIR = "/gpfs/helios/home/abdelrah/ModernBERT/Benchmarking/benchmark-scripts/sa-benchmarking/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename
log_filename = get_log_filename(args.model_name, args.epochs, PATIENCE, args.dataset_name)
log_filepath = os.path.join(LOG_DIR, log_filename)

# Set up logging with the dynamic filename
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_filepath)],
    force=True,
)
logging.info(f"Starting benchmarking for model {args.model_name} on dataset {args.dataset_name}")
logging.info(f"Configuration: epochs={args.epochs}, patience={PATIENCE}, batch_size={args.batch_size}")
print(f"Logging to {log_filepath}")

PARENT_PATH = "/gpfs/helios/home/abdelrah/ModernBERT/Benchmarking/Data/"
MODERN_BERT_TOKENIZER_PATH = "/gpfs/helios/home/abdelrah/ModernBERT/Training2/Tokenizer"

if args.dataset_name.lower() == "hard":
    DATA_DIR = f"{PARENT_PATH}/hard"
elif args.dataset_name.lower() in ["astd", "labr", "ajgt"]:
    DATA_DIR = f"{PARENT_PATH}/{args.dataset_name.lower()}"
else:
    DATA_DIR = f"{PARENT_PATH}/{args.dataset_name.lower()}"

if args.dataset_name.lower() == "labr":
    TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
    TEST_FILE = os.path.join(DATA_DIR, "test.txt")
    EVAL_FILE = TEST_FILE
else:
    TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
    VAL_FILE = os.path.join(DATA_DIR, "validation.txt")
    TEST_FILE = os.path.join(DATA_DIR, "test.txt")
    EVAL_FILE = TEST_FILE if args.benchmark else VAL_FILE

datasets_dict = {
    "hard": {
        "name": "Elnagara/hard",
        "num_labels": 2,
        "load_type": "hf",
        "token": HF_TOKEN
    },
    "astd": {
        "name": "ASTD",
        "num_labels": 4,
        "load_type": "csv",
        "url": "https://raw.githubusercontent.com/mahmoudnabil/ASTD/master/data/Tweets.txt?raw=true",
        "benchmark_train": "https://raw.githubusercontent.com/mahmoudnabil/ASTD/master/data/4class-balanced-train.txt?raw=true",
        "benchmark_test": "https://raw.githubusercontent.com/mahmoudnabil/ASTD/master/data/4class-balanced-test.txt?raw=true",
        "benchmark_validation": "https://raw.githubusercontent.com/mahmoudnabil/ASTD/master/data/4class-balanced-validation.txt?raw=true",
        "column_names": ["text", "label"]
    },
    "labr": {
        "name": "LABR",
        "num_labels": 2,
        "load_type": "csv",
        "url": "https://raw.githubusercontent.com/mohamedadaly/LABR/master/data/reviews.tsv?raw=true",
        "benchmark_train": "https://raw.githubusercontent.com/mohamedadaly/LABR/master/data/2class-unbalanced-train.txt?raw=true",
        "benchmark_test": "https://raw.githubusercontent.com/mohamedadaly/LABR/master/data/2class-unbalanced-test.txt?raw=true",
        "column_names": ["rating", "review id", "user_id", "book_id", "review"]
    },
    "ajgt": {
        "name": "AJGT",
        "num_labels": 2,
        "load_type": "xlsx",
        "url": "https://github.com/komari6/Arabic-twitter-corpus-AJGT/blob/master/AJGT.xlsx?raw=true",
        "column_names": ["text", "label"]
    }
}

models = {
    "modernbert": "/gpfs/helios/home/abdelrah/ModernBERT/Training2/output/checkpoint_step_13000/",
    "arabert": "aubmindlab/bert-base-arabert",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "arabert2": "aubmindlab/bert-base-arabertv2",
    "marbert": "UBC-NLP/MARBERTv2",
}

def get_memory_usage():
    """
    Get current RAM and VRAM usage statistics.
    
    Returns:
        Dict containing memory usage information:
            - ram_used_gb: RAM currently in use (GB)
            - ram_percent: Percentage of total RAM in use
            - vram_used_gb: VRAM currently in use (GB)
            - vram_total_gb: Total VRAM available (GB)
            - vram_percent: Percentage of VRAM in use
    """
    memory_stats = {}
    
    # Get RAM usage
    process = psutil.Process(os.getpid())
    ram_used_bytes = process.memory_info().rss  # Resident Set Size
    ram_used_gb = ram_used_bytes / (1024 ** 3)
    memory_stats["ram_used_gb"] = ram_used_gb
    memory_stats["ram_percent"] = psutil.virtual_memory().percent
    
    # Get VRAM usage if CUDA is available
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            vram_used_bytes = torch.cuda.memory_allocated()
            vram_reserved_bytes = torch.cuda.memory_reserved()
            vram_total_bytes = torch.cuda.get_device_properties(0).total_memory
            
            vram_used_gb = vram_used_bytes / (1024 ** 3)
            vram_total_gb = vram_total_bytes / (1024 ** 3)
            vram_percent = (vram_used_bytes / vram_total_bytes) * 100
            
            memory_stats["vram_used_gb"] = vram_used_gb
            memory_stats["vram_total_gb"] = vram_total_gb
            memory_stats["vram_percent"] = vram_percent
        except Exception as e:
            logging.warning(f"Could not get VRAM usage: {e}")
            memory_stats["vram_used_gb"] = 0.0
            memory_stats["vram_total_gb"] = 0.0
            memory_stats["vram_percent"] = 0.0
    else:
        memory_stats["vram_used_gb"] = 0.0
        memory_stats["vram_total_gb"] = 0.0
        memory_stats["vram_percent"] = 0.0
    
    return memory_stats

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    chosen_dataset = args.dataset_name.lower()
    if chosen_dataset == "hard":
        DATA_DIR = f"{PARENT_PATH}/hard"
        dataset = load_dataset("Elnagara/hard", "plain_text",
                               split="train", token=HF_TOKEN)
        TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
        VAL_FILE = os.path.join(DATA_DIR, "validation.txt")
        TEST_FILE = os.path.join(DATA_DIR, "test.txt")
        if not (os.path.exists(TRAIN_FILE) and os.path.exists(VAL_FILE) and os.path.exists(TEST_FILE)):
            process_dataset(dataset, window_size=8192, base_dir=DATA_DIR)
    elif chosen_dataset == "astd":
        DATA_DIR = f"{PARENT_PATH}/astd"
        TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
        VAL_FILE = os.path.join(DATA_DIR, "validation.txt")
        TEST_FILE = os.path.join(DATA_DIR, "test.txt")
        if not (os.path.exists(TRAIN_FILE) and os.path.exists(VAL_FILE) and os.path.exists(TEST_FILE)):
            prepare_astd_benchmark(DATA_DIR, datasets_dict["astd"])
        dataset = None
    elif chosen_dataset == "labr":
        DATA_DIR = f"{PARENT_PATH}/labr"
        prepare_labr_benchmark(DATA_DIR, datasets_dict["labr"])
        TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
        VAL_FILE = None
        TEST_FILE = os.path.join(DATA_DIR, "test.txt")
        EVAL_FILE = TEST_FILE
        dataset = None
    elif chosen_dataset == "ajgt":
        DATA_DIR = f"{PARENT_PATH}/ajgt"
        df = pd.read_excel(datasets_dict["ajgt"]["url"], engine="openpyxl")
        if len(df.columns) == 3:
            df = df.iloc[:, 1:3]
        else:
            df = df.iloc[:, :2]
        df.columns = datasets_dict["ajgt"]["column_names"]
        ds = Dataset.from_pandas(df)
        TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
        VAL_FILE = os.path.join(DATA_DIR, "validation.txt")
        TEST_FILE = os.path.join(DATA_DIR, "test.txt")
        if not (os.path.exists(TRAIN_FILE) and os.path.exists(VAL_FILE) and os.path.exists(TEST_FILE)):
            process_dataset(ds, window_size=8192, base_dir=DATA_DIR)
        dataset = ds
    else:
        raise ValueError("Unknown dataset selected.")

    if chosen_dataset not in ["astd", "labr"] and not args.use_streaming:
        if not args.preprocess_flag:
            process_dataset(dataset, window_size=8192, base_dir=DATA_DIR)
    elif args.use_streaming:
        print("Streaming mode enabled. Skipping in-memory processing and splitting.")
    
    # if args.inference_flag and args.checkpoint is not None:
    #    EVAL_FILE = VAL_FILE if chosen_dataset != "labr" else TEST_FILE
    #else:
    #    EVAL_FILE = TEST_FILE if args.benchmark else (
    #        VAL_FILE if chosen_dataset != "labr" else TEST_FILE)

    def main():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        print(f"Using device: {device}")

        benchmark_results = {}
        model_predictions = {}
        model_confidences = {}
        model_true_labels = None

        model_name = args.model_name
        model_path = models.get(model_name)
        logging.info(f"Benchmarking model: {model_name}")
        print(f"\nBenchmarking model: {model_name}")

        tokenizer_path = MODERN_BERT_TOKENIZER_PATH if model_name == "modernbert" else model_path

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, token=HF_TOKEN)
        config = AutoConfig.from_pretrained(
            model_path, num_labels=datasets_dict[args.dataset_name]['num_labels'], token=HF_TOKEN)
        
        model = AutoModelForSequenceClassification.from_config(
            config) 
        
        hidden_size = model.config.hidden_size
        model.classifier = nn.Linear(
            hidden_size, datasets_dict[args.dataset_name]['num_labels'])

        if args.freeze:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        model.to(device)

        extra_kwargs = {}
        if args.dataset_name.lower() == "labr":
            extra_kwargs["dataset_type"] = "labr"
        elif args.dataset_name.lower() == "ajgt":
            extra_kwargs["dataset_type"] = "ajgt"
        elif args.dataset_name.lower() == "astd":
            extra_kwargs["dataset_type"] = "astd"
        else:
            extra_kwargs["dataset_type"] = "default"

        train_samples = load_sentiment_dataset(
            TRAIN_FILE,
            tokenizer,
            None,
            num_labels=datasets_dict[args.dataset_name]['num_labels'],
            max_length=args.max_size,
            dataset_type=extra_kwargs.get("dataset_type", "default")
        )
        if len(train_samples) == 0:
            logging.error(
                f"No training samples loaded for model {model_name}.")
            raise Exception("No training samples.")
        train_dataloader = DataLoader(
            train_samples,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
        if args.dataset_name.lower() == "labr":
            logging.warning(
                "No validation samples loaded for LABR dataset; using training samples as validation.")
            val_dataloader = None
        else:
            val_samples = load_sentiment_dataset(
                VAL_FILE,
                tokenizer,
                None,
                num_labels=datasets_dict[args.dataset_name]['num_labels'],
                max_length=args.max_size,
                dataset_type=extra_kwargs.get("dataset_type", "default")
            )
            if len(val_samples) == 0:
            
                logging.error(
                    f"No validation samples loaded for model {model_name}.")
                raise Exception("No validation samples.")
            val_dataloader = DataLoader(
                val_samples,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )

        logging.info(f"Starting training for model: {model_name}")
        
        # Log initial memory usage
        initial_memory = get_memory_usage()
        logging.info("Initial memory usage before training:")
        logging.info(f"  RAM: {initial_memory['ram_used_gb']:.2f} GB ({initial_memory['ram_percent']:.1f}%)")
        if torch.cuda.is_available():
            logging.info(f"  VRAM: {initial_memory['vram_used_gb']:.2f} GB ({initial_memory['vram_percent']:.1f}%)")
        
        model = train_model(
            model,
            train_dataloader,
            val_dataloader,
            device,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=PATIENCE,  # Use the defined PATIENCE constant instead of hardcoded value
            checkpoint_path=args.checkpoint,
            continue_from_checkpoint=args.continue_from_checkpoint,
            save_every=args.save_every
        )
        
        # Log final memory usage after training
        final_memory = get_memory_usage()
        logging.info("Final memory usage after training:")
        logging.info(f"  RAM: {final_memory['ram_used_gb']:.2f} GB ({final_memory['ram_percent']:.1f}%)")
        if torch.cuda.is_available():
            logging.info(f"  VRAM: {final_memory['vram_used_gb']:.2f} GB ({final_memory['vram_percent']:.1f}%)")
        
        test_samples = load_sentiment_dataset(
                TEST_FILE,
              tokenizer,
              None,
              num_labels=datasets_dict[args.dataset_name]['num_labels'],
              max_length=args.max_size,
              dataset_type=extra_kwargs.get("dataset_type", "default")
          )

        test_dataloader = DataLoader(
            test_samples,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        macro_f1, report, preds, true_labels, confidences, avg_eval_loss, perplexity = evaluate_model(
            model, test_dataloader, device)
        benchmark_results[model_name] = {
            "macro_f1": macro_f1,
            "report": report,
            "avg_eval_loss": avg_eval_loss,
            "perplexity": perplexity
        }
        model_predictions[model_name] = preds
        model_confidences[model_name] = confidences
        if model_true_labels is None:
            model_true_labels = true_labels

        logging.info(f"{model_name} Evaluation Macro-F1: {macro_f1:.4f}")
        logging.info(f"{model_name} Classification Report:\n{report}")
        logging.info(
            f"{model_name} Average Eval Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.4f}")
        print(f"{model_name} Evaluation Macro-F1: {macro_f1:.4f}")
        print(f"{model_name} Classification Report:")
        print(report)
        print(
            f"{model_name} Average Evaluation Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.4f}")

        print(
            f"\nComparison of ground truth, predictions, and confidences for {model_name}:")
        print(f"{'Index':<6}{'Ground Truth':<15}{'Prediction':<15}{'Confidence':<12}")
        for i in range(min(10, len(true_labels))):
            print(
                f"{i:<6}{true_labels[i]:<15}{preds[i]:<15}{confidences[i]:<12.4f}")
        print("-" * 40)

        if args.checkpoint is not None:
            torch.save(model.state_dict(), args.checkpoint)
            tokenizer.save_pretrained(os.path.dirname(args.checkpoint))
            logging.info(
                f"Final model and tokenizer saved to {args.checkpoint} and its directory.")

        torch.cuda.empty_cache()

        # Save benchmark results to a JSON file with the same base name as the log
        result_filepath = log_filepath.replace('.log', '_results.json')
        with open(result_filepath, 'w') as f:
            json.dump({
                "configuration": {
                    "model": model_name,
                    "dataset": args.dataset_name,
                    "epochs": args.epochs,
                    "patience": PATIENCE,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "max_sequence_length": args.max_size,
                    "freeze_layers": args.freeze
                },
                "hardware_usage": {
                    "initial_memory": {
                        "ram_used_gb": float(initial_memory['ram_used_gb']),
                        "ram_percent": float(initial_memory['ram_percent']),
                        "vram_used_gb": float(initial_memory['vram_used_gb']),
                        "vram_total_gb": float(initial_memory['vram_total_gb']),
                        "vram_percent": float(initial_memory['vram_percent'])
                    },
                    "final_memory": {
                        "ram_used_gb": float(final_memory['ram_used_gb']),
                        "ram_percent": float(final_memory['ram_percent']),
                        "vram_used_gb": float(final_memory['vram_used_gb']),
                        "vram_total_gb": float(final_memory['vram_total_gb']),
                        "vram_percent": float(final_memory['vram_percent'])
                    }
                },
                "results": {
                    model_name: {
                        "macro_f1": float(macro_f1),
                        "avg_eval_loss": float(avg_eval_loss),
                        "perplexity": float(perplexity)
                    }
                }
            }, f, indent=2)
        logging.info(f"Results saved to {result_filepath}")
        print(f"Detailed results saved to {result_filepath}")

        print("\nBenchmarking Complete. Summary of Results:")
        for model_name, metrics in benchmark_results.items():
            print(
                f"{model_name}: Macro-F1 = {metrics['macro_f1']:.4f} | Perplexity = {metrics['perplexity']:.4f}")
        
        print("\nHardware Usage Summary:")
        print(f"Initial RAM: {initial_memory['ram_used_gb']:.2f} GB ({initial_memory['ram_percent']:.1f}%)")
        print(f"Final RAM: {final_memory['ram_used_gb']:.2f} GB ({final_memory['ram_percent']:.1f}%)")
        if torch.cuda.is_available():
            print(f"Initial VRAM: {initial_memory['vram_used_gb']:.2f} GB ({initial_memory['vram_percent']:.1f}%)")
            print(f"Final VRAM: {final_memory['vram_used_gb']:.2f} GB ({final_memory['vram_percent']:.1f}%)")
    # def infere():
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     logging.info(f"Using device: {device}")
    #     print(f"Using device: {device}")
    # 
    #     benchmark_results = {}
    #     model_predictions = {}
    #     model_confidences = {}
    #     model_true_labels = None
    # 
    #     model_name = args.model_name
    #     model_path = args.checkpoint
    #     logging.info(f"Benchmarking model: {model_name}")
    #     print(f"\nBenchmarking model: {model_name}")
    # 
    #     tokenizer_path = model_path if model_name == "arabert" else MODERN_BERT_TOKENIZER_PATH
    # 
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         tokenizer_path, token=HF_TOKEN)
    #     model = AutoModelForSequenceClassification.from_pretrained(
    #         model_path, token=HF_TOKEN)
    # 
    #     config = AutoConfig.from_pretrained(
    #         model_path, num_labels=datasets_dict[args.dataset_name]['num_labels'], token=HF_TOKEN)
    #     hidden_size = model.config.hidden_size
    #     model.classifier = nn.Linear(
    #         hidden_size, datasets_dict[args.dataset_name]['num_labels'])
    # 
    #     val_samples = load_sentiment_dataset(
    #         EVAL_FILE,
    #         tokenizer,
    #         None,
    #         num_labels=datasets_dict[args.dataset_name]['num_labels'],
    #         max_length=args.max_size,
    #         dataset_type=extra_kwargs.get("dataset_type", "default")
    #     )
    #     if len(val_samples) == 0:
    #         if args.dataset_name.lower() == "labr":
    #             logging.warning(
    #                 "No validation samples loaded for LABR dataset; using training samples as validation.")
    #             val_samples = train_samples
    #         else:
    #             logging.error(
    #                 f"No validation samples loaded for model {model_name}.")
    #             raise Exception("No validation samples.")
    #     val_dataloader = DataLoader(
    #         val_samples,
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         num_workers=args.num_workers,
    #         pin_memory=True
    #     )
    # 
    #     if checkpoint_path is not None and os.path.exists(checkpoint_path):
    #         logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    #         state = torch.load(checkpoint_path, map_location=device)
    #         model.load_state_dict(state)
    #     else:
    #         raise Exception("Please provide a checkpoint path to use for inference.")
    # 
    #     logging.info(f"Starting inference for model: {model_name}")
    #     macro_f1, report, preds, true_labels, confidences, avg_eval_loss, perplexity = evaluate_model(
    #         model, val_dataloader, device)
    #     benchmark_results[model_name] = {
    #         "macro_f1": macro_f1,
    #         "report": report,
    #         "avg_eval_loss": avg_eval_loss,
    #         "perplexity": perplexity
    #     }
    #     model_predictions[model_name] = preds
    #     model_confidences[model_name] = confidences
    #     if model_true_labels is None:
    #         model_true_labels = true_labels
    # 
    #     logging.info(f"{model_name} Evaluation Macro-F1: {macro_f1:.4f}")
    #     logging.info(f"{model_name} Classification Report:\n{report}")
    #     logging.info(
    #         f"{model_name} Average Eval Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.4f}")
    #     print(f"{model_name} Evaluation Macro-F1: {macro_f1:.4f}")
    #     print(f"{model_name} Classification Report:")
    #     print(report)
    #     print(
    #         f"{model_name} Average Evaluation Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.4f}")
    # 
    #     print(
    #         f"\nComparison of ground truth, predictions, and confidences for {model_name}:")
    #     print(f"{'Index':<6}{'Ground Truth':<15}{'Prediction':<15}{'Confidence':<12}")
    #     for i in range(min(10, len(true_labels))):
    #         print(
    #             f"{i:<6}{true_labels[i]:<15}{preds[i]:<15}{confidences[i]:<12.4f}")
    #     print("-" * 40)
    # 
    #     # Save inference results to a JSON file with the same base name as the log
    #     result_filepath = log_filepath.replace('.log', '_results.json')
    #     with open(result_filepath, 'w') as f:
    #         json.dump({
    #             "configuration": {
    #                 "model": model_name,
    #                 "dataset": args.dataset_name,
    #                 "epochs": args.epochs,
    #                 "patience": PATIENCE,
    #                 "batch_size": args.batch_size,
    #                 "learning_rate": args.learning_rate,
    #                 "max_sequence_length": args.max_size,
    #                 "freeze_layers": args.freeze
    #             },
    #             "results": {
    #                 model_name: {
    #                     "macro_f1": float(macro_f1),
    #                     "avg_eval_loss": float(avg_eval_loss),
    #                     "perplexity": float(perplexity)
    #                 }
    #             }
    #         }, f, indent=2)
    #     logging.info(f"Results saved to {result_filepath}")
    #     print(f"Detailed results saved to {result_filepath}")
    # 
    #     print("\nBenchmarking Complete. Summary of Results:")
    #     for model_name, metrics in benchmark_results.items():
    #         print(
    #             f"{model_name}: Macro-F1 = {metrics['macro_f1']:.4f} | Perplexity = {metrics['perplexity']:.4f}")
        

    main()
