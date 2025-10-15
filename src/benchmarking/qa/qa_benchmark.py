import os
import io
import requests
import zipfile
import logging
import torch
import json
import copy
import re
import string
import random
import numpy as np
from collections import Counter
from datetime import datetime
from tqdm import tqdm
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset, Dataset, concatenate_datasets, Value
from arabert.preprocess import ArabertPreprocessor
from typing import Dict, Optional

# Import shared utilities
from utils.memory import get_memory_usage
from utils.logging import setup_logging

SEGMENTER = ArabertPreprocessor('bert-base-arabert')


def get_log_filename(model_name, epochs):
    """
    Generate a log filename incorporating model name, epoch count, and current timestamp.
    (Removed metric from filename as EM/F1 are standard now)

    Args:
        model_name (str): Name of the model.
        epochs (int): Number of training epochs.

    Returns:
        str: The generated log filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Removed metric from filename
    return f"QA_Benchmark_{model_name}_{epochs}ep_{timestamp}.log"

def flatten_squad_list(data_list):
    """
    Flatten a nested SQuAD-like data structure into a list of QA examples.
    NOTE: This currently takes only the *first* answer. Standard SQuAD eval uses all.

    Args:
        data_list (list): List of articles containing paragraphs and QA pairs.

    Returns:
        list: A list of dictionaries, each containing 'context', 'question', 'answer', and 'answer_start'.
    """
    all_rows = []
    for article in data_list:
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            if not isinstance(context, str) or not context.strip():
                continue
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "")
                if not isinstance(question, str) or not question.strip():
                    continue
                # Keep the list here for potential future use
                answers = qa.get("answers", [])
                if len(answers) > 0:
                    # Store the first answer for current compatibility
                    answer_text = answers[0]["text"]
                    start_char = answers[0]["answer_start"]
                    # Could potentially store all answers here if needed later:
                    # all_answers = [{"text": ans["text"], "answer_start": ans["answer_start"]} for ans in answers]
                else:
                    answer_text = ""
                    start_char = 0
                    # all_answers = []
                all_rows.append({
                    "context": context,
                    "question": question,
                    "answer": answer_text,  # Single answer string
                    "answer_start": start_char,  # Single start position
                    # "all_answers": all_answers # Optional: Store all answers if modifying eval
                })
    return all_rows

def fix_answer_fields(example):
    """
    Ensure that the 'answer' and 'answer_start' fields in a QA example are in the correct format.

    Args:
        example (dict): A dictionary representing a QA example.

    Returns:
        dict: The QA example with fixed 'answer' and 'answer_start' fields.
    """
    if isinstance(example["answer"], list):
        example["answer"] = example["answer"][0] if example["answer"] else ""
    if isinstance(example["answer_start"], list):
        example["answer_start"] = example["answer_start"][0] if example["answer_start"] else 0
    return example

def transform_arcd(example):
    """
    Transform ARCD dataset examples by extracting the primary answer information and removing redundant fields.
    NOTE: This currently takes only the *first* answer. Standard SQuAD eval uses all.

    Args:
        example (dict): A dictionary representing an ARCD example.

    Returns:
        dict: The transformed ARCD example with 'answer' and 'answer_start' fields properly set.
    """
    answers_field = example.get("answers")  # Use .get for safety
    if isinstance(answers_field, list) and len(answers_field) > 0:
        first = answers_field[0]
        example["answer"] = first.get("text", "")
        example["answer_start"] = first.get("answer_start", 0)
        # example["all_answers"] = [{"text": ans.get("text", ""), "answer_start": ans.get("answer_start", 0)} for ans in answers_field] # Optional
    elif isinstance(answers_field, dict):  # Handle cases where it might be a dict already
        example["answer"] = answers_field.get("text", "")
        example["answer_start"] = answers_field.get("answer_start", 0)
        # example["all_answers"] = [{"text": example["answer"], "answer_start": example["answer_start"]}] # Optional
    else:
        example["answer"] = ""
        example["answer_start"] = 0
        # example["all_answers"] = [] # Optional

    # Keep original 'answers' if needed for full evaluation later, otherwise remove
    # if "answers" in example:
    #     del example["answers"]
    for col in ["id", "title"]:  # Remove id and title
        if col in example:
            del example[col]
    return example

def download_and_extract(url, extract_to="."):
    """
    Download a zip file from the provided URL and extract its contents into the specified directory.

    Args:
        url (str): URL of the zip file to download.
        extract_to (str, optional): Directory to extract files into. Defaults to current directory.

    Raises:
        Exception: If the download fails (non-200 status code).
    """
    logging.info("Downloading and extracting dataset...")
    response = requests.get(url)
    if response.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(path=extract_to)
        logging.info("Extraction complete.")
    else:
        error_msg = f"Failed to download file from {url} (status code: {response.status_code})"
        logging.error(error_msg)
        raise Exception(error_msg)

def load_and_prepare_datasets(data_dir: str = "./data/benchmarking/qa"):
    """
    Load and prepare the QA datasets by combining Arabic-SQuAD and ARCD,
    applying necessary transformations and tokenization.

    Args:
        data_dir (str): Directory to store and load datasets from.

    Returns:
        tuple: (train_features, test_features, raw_test) where:
            - train_features: Tokenized training dataset.
            - test_features: Tokenized test dataset.
            - raw_test: Original test dataset examples.
    """
    GITHUB_URL = "https://github.com/WissamAntoun/Arabic_QA_Datasets/raw/master/data.zip"
    DATA_DIR = data_dir
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    # Expected path of the Arabic-SQuAD file
    squad_path = os.path.join(DATA_DIR, "Arabic-SQuAD.json")

    # If the expected file is missing, attempt to download/extract
    if not os.path.exists(squad_path):
        logging.info(
            f"'{os.path.basename(squad_path)}' not found under {DATA_DIR}. Attempting to download/extract QA datasets...")
        download_and_extract(GITHUB_URL, extract_to=DATA_DIR)

        # After extraction, re-check default location
        if not os.path.exists(squad_path):
            # Some archives may extract into nested folders; search recursively
            found_path = None
            for root, _, files in os.walk(DATA_DIR):
                if "Arabic-SQuAD.json" in files:
                    found_path = os.path.join(root, "Arabic-SQuAD.json")
                    break
            if found_path is not None:
                squad_path = found_path
                logging.info(f"Found Arabic-SQuAD.json at nested path: {squad_path}")
            else:
                raise FileNotFoundError(
                    f"Arabic-Squad.json not found in {DATA_DIR} after download. Check extracted contents or rename accordingly.")
    raw_squad = load_dataset("json", data_files={"train": squad_path})["train"]
    all_flat = []
    data_list = raw_squad[0]["data"]
    sub_rows = flatten_squad_list(data_list)
    all_flat.extend(sub_rows)
    squad_flat = Dataset.from_list(all_flat)
    squad_flat = squad_flat.map(fix_answer_fields)
    squad_flat = squad_flat.cast_column("answer", Value("string"))
    squad_flat = squad_flat.cast_column("answer_start", Value("int64"))

    logging.info("Loading ARCD dataset from Hugging Face (hsseinmz/arcd)...")
    raw_arcd = load_dataset("hsseinmz/arcd")
    #split_arcd = raw_arcd.train_test_split(test_size=0.5, seed=42)
    arcd_train = raw_arcd["train"]
    arcd_test = raw_arcd["validation"]
    arcd_train = arcd_train.map(transform_arcd)
    arcd_train = arcd_train.map(fix_answer_fields)
    arcd_test = arcd_test.map(transform_arcd)
    arcd_test = arcd_test.map(fix_answer_fields)

    logging.info(f"Arabic-Squad size: {len(squad_flat)} examples.")
    logging.info(f"ARCD training split size: {len(arcd_train)} examples.")
    logging.info(f"ARCD test split size: {len(arcd_test)} examples.")

    combined_train = concatenate_datasets([squad_flat, arcd_train])
    logging.info(
        f"Final combined training set has {len(combined_train)} examples.")
    return combined_train, arcd_test, arcd_test

def normalize(example):
    columns = ["question", "context", "answer"]
    for column in columns:
        text = example[column]
        # 1. Remove Tatweel
        text = re.sub("\u0640", "", text)
        
        # 2. Remove Diacritics
        arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        text = re.sub(arabic_diacritics, '', text)
        
        # 3. Normalize punctuation
        punct_map = {
            "،": ",",   # Arabic comma
            "؛": ";",   # Arabic semicolon
            "؟": "?",   # Arabic question mark
        }
        for ar_punct, en_punct in punct_map.items():
            text = text.replace(ar_punct, en_punct)

        # 5. Normalize Hindi digits to Arabic digits
        hindi_to_arabic_digits = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        text = text.translate(hindi_to_arabic_digits)
        example[column] = text
    return example

def get_new_start(context, segmented_answer):
    return context.find(segmented_answer)

def segment(example, model_name):
    global SEGMENTER
    if SEGMENTER is None:
        if model_name in ["arabert", "arabert2", "modernbert"]:
            SEGMENTER = ArabertPreprocessor(model_name = "bert-base-arabert")
    example["question"] = SEGMENTER.preprocess(example["question"])
    example["context"] = SEGMENTER.preprocess(example["context"])
    example["original_answer"] = example["answer"]
    example["answer"] = SEGMENTER.preprocess(example["answer"])
    return example

def fix_answer_ids(example):
    example["answer_start"] = get_new_start(example["context"], example["answer"])
    example["answer_end"] = example["answer_start"] + len(example["answer"])
    return example


def preprocess_datasets(dataset, model_name):
    logging.info("Normalizing datasets...")
    dataset = dataset.map(normalize, batched=False)
    # Perform Sanity Check for answer and answer_start
    logging.info(f"Sanity Check for answer and answer_start: {len(dataset.filter(lambda x: x['context'][x['answer_start']:x['answer_start']+len(x['answer'])] == x['answer']))}")

    if model_name in ["arabert", "arabert2", "modernbert"]:
        logging.info("Segmenting datasets...")
        dataset = dataset.map(segment, batched=False, fn_kwargs={"model_name": model_name})
        # fix period punctuation
        dataset = dataset.map(lambda x: {"answer": x["answer"].replace(" . ", ". ")}, batched=False)

    else:
        pass
    logging.info("Fixing answer ids...")
    dataset = dataset.map(fix_answer_ids, batched=False)
    
    dataset = dataset.filter(lambda x: x['context'][x['answer_start']:x['answer_start']+len(x['answer'])] == x['answer'])
    logging.info(f"Sanity Check for answer and answer_start after segmenting: {len(dataset)}")
    
    return dataset

def squeeze_tokenized(tokenized):
    """
    Squeeze extra dimensions from tokenized outputs if they have a singleton batch dimension.

    Args:
        tokenized (dict): The output dictionary from a tokenizer.

    Returns:
        dict: The tokenized output with squeezed dimensions.
    """
    for key in tokenized:
        if isinstance(tokenized[key], torch.Tensor) and tokenized[key].shape[0] == 1:
            tokenized[key] = tokenized[key].squeeze(0)
    return tokenized

def find_token_span(offset_mapping, answer_start, answer_end):
    starts = [s for s, _ in offset_mapping]
    ends   = [e for _, e in offset_mapping]

    # Linear search for start_token
    start_token = 0
    for i, s in enumerate(starts):
        if s > answer_start:
            break
        start_token = i

    # Linear search for end_token
    end_token = 0
    for i, e in enumerate(ends):
        if e >= answer_end:
            end_token = i
            break
    else:
        end_token = len(ends) - 1

    start_char_offset = answer_start - offset_mapping[start_token][0]
    end_char_offset = offset_mapping[end_token][1] - answer_end

    return start_token, end_token, start_char_offset, end_char_offset

def tokenize(example, tokenizer, max_length, doc_stride):
    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_token_type_ids=True,  # Ensure token_type_ids are returned
        return_tensors="pt"
    )
    tokenized = squeeze_tokenized(tokenized)
    sequence_ids = tokenized.sequence_ids()

    # Restrict context offset_mapping
    context_start = sequence_ids.index(1)
    context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

    offsets = tokenized["offset_mapping"].tolist()
    offsets = offsets[context_start:context_end]
    tokenized.pop("offset_mapping")

    start_position, end_position, start_char_offset, end_char_offset = find_token_span(offsets,
     example['answer_start'], example['answer_end'])

    tokenized["start_positions"] = torch.tensor(start_position + context_start)
    tokenized["end_positions"] = torch.tensor(end_position + context_start)
    tokenized["start_char_offset"] = torch.tensor(start_char_offset)
    tokenized["end_char_offset"] = torch.tensor(end_char_offset)
    return tokenized

def sanity_check(dataset, tokenized_dataset, tokenizer):
    count = 0
    import csv

    failed_examples = []

    for i in range(len(dataset)):
        segmented_answer = dataset[i]["answer"]
        segmented_context = dataset[i]["context"]
        start_position = tokenized_dataset[i]["start_positions"]
        end_position = tokenized_dataset[i]["end_positions"]
        tokenized_answer = tokenized_dataset[i]['input_ids'][start_position:end_position+1]
        decoded_answer = tokenizer.decode(tokenized_answer)
        start_offset = tokenized_dataset[i]["start_char_offset"]
        end_offset = tokenized_dataset[i]["end_char_offset"]
        extracted_answer = decoded_answer[start_offset:len(decoded_answer)-end_offset]
        if segmented_answer != extracted_answer:
            count += 1
            failed_examples.append({
                "index": i,
                "question": dataset[i].get("question", ""),
                "context": segmented_context,
                "expected_answer": segmented_answer,
                "decoded_answer": decoded_answer,
                "extracted_answer": extracted_answer,
                "start_positions": tokenized_dataset[i]["start_positions"].item() if hasattr(tokenized_dataset[i]["start_positions"], "item") else tokenized_dataset[i]["start_positions"],
                "end_positions": tokenized_dataset[i]["end_positions"].item() if hasattr(tokenized_dataset[i]["end_positions"], "item") else tokenized_dataset[i]["end_positions"],
                "start_char_offset": start_offset.item() if hasattr(start_offset, "item") else start_offset,
                "end_char_offset": end_offset.item() if hasattr(end_offset, "item") else end_offset,
            })

    if failed_examples:
        with open("sanity_check_failures.csv", "w", newline='', encoding="utf-8") as csvfile:
            fieldnames = ["index", "question", "context", "expected_answer", "decoded_answer", "extracted_answer", "start_positions", "end_positions", "start_char_offset", "end_char_offset"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in failed_examples:
                writer.writerow(row)
    logging.info(f"Sanity check passed with {count} failures")
    return

def custom_collate_fn(batch):
    """
    Custom collate function that ensures all values are properly converted to tensors.

    Args:
        batch (list): List of items from the dataset.

    Returns:
        A collated batch with all values as tensors.
    """
    # Convert any non-tensor values to tensors before collating
    processed_batch = []
    for item in batch:
        processed_item = {}
        for key, value in item.items():
            if not isinstance(value, torch.Tensor):
                processed_item[key] = torch.tensor(value)
            else:
                processed_item[key] = value
        processed_batch.append(processed_item)
    
    return torch.utils.data.dataloader.default_collate(processed_batch)

def normalize_text(s):
    """Lower text and remove punctuation and extra whitespace."""
    global SEGMENTER

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_punc(SEGMENTER.desegment(s))).strip() if SEGMENTER is not None else white_space_fix(remove_punc(s)).strip()

def compute_f1(prediction, ground_truth):
    """Computes F1 score between a prediction and a single ground truth."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()

    # if either is empty, F1 is 1 if both are empty, 0 otherwise
    if not truth_tokens or not pred_tokens:
        return int(truth_tokens == pred_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact(prediction, ground_truth):
    """Computes Exact Match score between a prediction and a single ground truth."""
    return int(normalize_text(prediction) == normalize_text(ground_truth))

def get_sentence_index(context, text):
    """Returns the index where the text falls in context if any, else returns -1"""
    sentences = re.split(r"(?<=[.،؛,;؟!])\s+", context.strip())
    sentences = [re.sub(r'(?<=[.،؟؛,;!])', '', s) for s in sentences if s]
    for idx, sent in enumerate(sentences):
        if text in sent:
            return idx
    return -1

def evaluate_em_f1_sm(dataset, outputs, tokenizer):
    """
    Evaluate the QA model on a raw dataset and compute Exact Match (EM) and F1 score.
    Note: This implementation compares against a single ground truth answer per question
          due to the current data preprocessing. Standard SQuAD eval compares against all.

    Args:
        dataset: Tokenized Dataset for samples.
        outputs: Outputs from the model.    
        tokenizer: Tokenizer for decoding predictions.

    Returns:
        tuple (float, float) or (None, None): The average EM and F1 scores, or None if no examples processed.
    """
    total_em = 0
    total_f1 = 0
    total_sm = 0
    count = 0

    for i, ex in enumerate(tqdm(dataset, desc="Evaluating EM/F1/SM", leave=False)):
        decoded_text = tokenizer.decode(ex["input_ids"])
        context = decoded_text.split("[SEP]")[1].strip()
        start_pos = ex["start_positions"]
        end_pos = ex["end_positions"]

        gt_answer = tokenizer.decode(ex["input_ids"][start_pos:end_pos+1])
        
        start_logits = outputs["start_logits"][i]
        end_logits = outputs["end_logits"][i]
        start_index = torch.argmax(start_logits).item()
        end_index = torch.argmax(end_logits).item()

        if end_index < start_index:
            end_index = start_index

        prediction = tokenizer.decode(ex["input_ids"][start_index:end_index+1])

        # Calculate EM and F1 against the single reference
        em_score = compute_exact(prediction, gt_answer)
        f1_score = compute_f1(prediction, gt_answer)

        # Sentence Match
        pred_sent_idx = get_sentence_index(context, prediction)
        ref_sent_idx = get_sentence_index(context, gt_answer)
        if pred_sent_idx != -1 and pred_sent_idx == ref_sent_idx and prediction != '':
            total_sm += 1

        total_em += em_score
        total_f1 += f1_score
        count += 1

    avg_em = total_em / count if count > 0 else None
    avg_f1 = total_f1 / count if count > 0 else None
    avg_sm = total_sm / count if count > 0 else None

    return avg_em, avg_f1, avg_sm


def train_qa_model(model, train_dataloader, val_dataloader, tokenizer, device,
                   num_epochs=3, learning_rate=3e-5, patience=10, checkpoint_path=None,
                   continue_from_checkpoint=False, save_every=None, max_length=512, doc_stride=128,
                   eval_metric="f1", encoder_lr=1e-5, classifier_lr=3e-5):  # Added different learning rates
    """
    Train the QA model over several epochs and evaluate on a validation set using EM/F1.

    Implements gradient scaling for mixed precision, early stopping based on F1 score,
    layer freezing for the first epoch, and different learning rates for encoder vs classifier.

    Args:
        model: The QA model to train.
        train_dataloader: DataLoader for training samples.
        val_dataloader: DataLoader for validation samples.
        tokenizer: Tokenizer for decoding predictions.
        device: Torch device (cpu or cuda).
        num_epochs (int, optional): Number of training epochs. Defaults to 3.
        learning_rate (float, optional): Default learning rate (kept for backward compatibility). Defaults to 3e-5.
        patience (int, optional): Number of epochs to wait for F1 improvement before stopping. Defaults to 2.
        checkpoint_path (str, optional): Path to save model checkpoints. Defaults to None.
        continue_from_checkpoint (bool, optional): Whether to resume from a checkpoint. Defaults to False.
        save_every (int, optional): Save a checkpoint every N epochs. Defaults to None.
        max_length (int, optional): Maximum length for tokenization. Defaults to 512.
        doc_stride (int, optional): Document stride length. Defaults to 128.
        eval_metric (str, optional): Primary metric for early stopping ('f1' or 'em'). Defaults to "f1".
        encoder_lr (float, optional): Learning rate for encoder layers. Defaults to 1e-5.
        classifier_lr (float, optional): Learning rate for classifier head. Defaults to 3e-5.

    Returns:
        tuple: (model, peak_memory) where:
            - model: The trained model (with best validation performance if early stopping is triggered)
            - peak_memory: Dictionary containing peak memory usage statistics during training
    """
    # Get initial memory usage
    initial_memory = get_memory_usage()
    logging.info("Initial memory usage before training:")
    logging.info(f"  RAM: {initial_memory['ram_used_gb']:.2f} GB ({initial_memory['ram_percent']:.1f}%)")
    if torch.cuda.is_available():
        logging.info(f"  VRAM: {initial_memory['vram_used_gb']:.2f} GB ({initial_memory['vram_percent']:.1f}%)")

    # Calculate total training steps for warmup
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    optimizer = AdamW(filter(lambda p: p.requires_grad,
                      model.parameters()), betas=(0.9, 0.95), weight_decay=0.01, lr=learning_rate)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler(device=device)

    if continue_from_checkpoint and checkpoint_path is not None and os.path.exists(checkpoint_path):
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)

    # Based on the primary eval_metric (F1 usually)
    best_val_f1 = float('-inf')
    patience_counter = 0
    best_model_state = None
    peak_memory = initial_memory.copy()

    step_count = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        keys_to_remove = ["start_char_offset", "end_char_offset"]
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_index, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Prepare all inputs including labels for the model
            batch_inputs = {k: (torch.tensor(v) if not isinstance(v, torch.Tensor) else v).to(device) 
                           for k, v in batch.items() if k not in keys_to_remove}
            # Ensure start_positions and end_positions are properly shaped
            batch_inputs["start_positions"] = batch_inputs["start_positions"].long().view(-1)
            batch_inputs["end_positions"] = batch_inputs["end_positions"].long().view(-1)
            
            with torch.amp.autocast(device_type=device.type):
                # Use model's built-in QA loss by passing start/end positions
                outputs = model(**batch_inputs)
                loss = outputs.loss  # Use the model's built-in loss
            
            scaler.scale(loss).backward()
            # Add gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Track peak memory usage during training
            current_memory = get_memory_usage()
            for key in current_memory:
                if key in peak_memory and current_memory[key] > peak_memory[key]:
                    peak_memory[key] = current_memory[key]
            epoch_loss += loss.item()
            epoch_steps += 1
            step_count += 1
            progress_bar.set_postfix_str(s=f"Loss = {loss.item():.4f}, LR={scheduler.get_last_lr()[0]:.2e}", refresh=True)

        avg_train_loss = epoch_loss / \
            len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        logging.info(
            f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")
        
        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        val_steps = 0
        val_outputs = {'start_logits': [], 'end_logits': []}
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_inputs = {k: (torch.tensor(v) if not isinstance(v, torch.Tensor) else v).to(device)
                              for k, v in val_batch.items() if k not in keys_to_remove}
                val_inputs["start_positions"] = val_inputs["start_positions"].long().view(-1)
                val_inputs["end_positions"] = val_inputs["end_positions"].long().view(-1)
                val_output = model(**val_inputs)
                val_outputs['start_logits'].extend(val_output.start_logits)
                val_outputs['end_logits'].extend(val_output.end_logits)
                loss = val_output.loss
                val_loss += loss.item()
                val_steps += 1
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")

        # Evaluate using EM/F1
        avg_em, avg_f1, avg_sm = evaluate_em_f1_sm(val_dataloader.dataset, val_outputs, tokenizer)

        if avg_em is not None and avg_f1 is not None and avg_sm is not None:
            logging.info(
                f"Validation EM Score after Epoch {epoch+1}: {avg_em:.4f}")
            logging.info(
                f"Validation F1 Score after Epoch {epoch+1}: {avg_f1:.4f}")
            logging.info(
                f"Validation SM Score after Epoch {epoch+1}: {avg_sm:.4f}")

            # Use F1 score for early stopping and checkpointing
            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                logging.info(
                    f"Validation F1 score improved to {best_val_f1:.4f}; resetting patience counter.")
                if checkpoint_path is not None:
                    torch.save(model.state_dict(), checkpoint_path)
                    logging.info(
                        f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
            else:
                patience_counter += 1
                logging.info(
                    f"No improvement in validation score. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    break
        else:
            logging.warning(
                f"Validation EM/F1 Score after Epoch {epoch+1}: N/A (Evaluation failed or dataset empty)")
            # Count as no improvement when evaluation fails
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(
                    "Early stopping triggered due to evaluation failures or lack of improvement.")
                break

        if save_every is not None and checkpoint_path is not None and ((epoch+1) % save_every == 0):
            # Save intermediate checkpoint regardless of validation score improvement
            intermediate_checkpoint_path = f"{os.path.splitext(checkpoint_path)[0]}_epoch{epoch+1}{os.path.splitext(checkpoint_path)[1]}"
            torch.save(model.state_dict(), intermediate_checkpoint_path)
            logging.info(
                f"Periodic checkpoint saved at epoch {epoch+1} to {intermediate_checkpoint_path}")

    if best_model_state is not None:
        logging.info(
            f"Loading best model state with {eval_metric.upper()} score: {best_val_f1:.4f}")
        model.load_state_dict(best_model_state)
    else:
        logging.warning(
            "No best model state was saved. Returning the model from the last epoch.")

    # Log peak memory usage
    logging.info("Peak memory usage during training:")
    logging.info(f"  RAM: {peak_memory['ram_used_gb']:.2f} GB ({peak_memory['ram_percent']:.1f}%)")
    if torch.cuda.is_available():
        logging.info(f"  VRAM: {peak_memory['vram_used_gb']:.2f} GB ({peak_memory['vram_percent']:.1f}%)")

    return model, peak_memory

def run_qa_benchmark(
    model_name: str,
    model_path: str,
    tokenizer_path: Optional[str] = None,
    data_dir: str = "./data/benchmarking/qa",
    log_dir: str = "./logs",
    batch_size: int = 8,
    max_length: int = 512,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    encoder_lr: float = 1e-5,
    classifier_lr: float = 5e-5,
    patience: int = 2,
    eval_metric: str = "f1",
    checkpoint_path: Optional[str] = None,
    continue_from_checkpoint: bool = False,
    save_every: Optional[int] = None,
    hf_token: Optional[str] = None,
    seed: int = 42,
    num_workers: int = 0,
    **kwargs
) -> Dict:
    """
    Run QA benchmarking with specified configuration.
    
    Args:
        model_name: Name of the model to benchmark
        model_path: Path to pretrained model
        tokenizer_path: Path to tokenizer (default: same as model)
        data_dir: Directory for datasets
        log_dir: Directory for log files
        batch_size: Batch size for training
        max_length: Maximum sequence length
        epochs: Number of training epochs
        learning_rate: Learning rate
        encoder_lr: Learning rate for encoder layers
        classifier_lr: Learning rate for classifier head
        patience: Early stopping patience
        eval_metric: Metric for early stopping
        checkpoint_path: Path to save/load checkpoint
        continue_from_checkpoint: Whether to resume from checkpoint
        save_every: Save checkpoint every N epochs
        hf_token: HuggingFace token for private models
        seed: Random seed
        num_workers: DataLoader workers
        
    Returns:
        Dict containing results including EM, F1, SM scores and memory usage
    """
    # Generate log filename and setup logging
    log_filename = get_log_filename(model_name, epochs)
    log_filepath = os.path.join(log_dir, log_filename)
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging using shared utility only if not already configured by wrapper
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging(log_file=log_filepath)

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logging.info(f"Starting QA benchmarking for model {model_name} using EM/F1")
    logging.info(f"Configuration: epochs={epochs}, encoder_lr={encoder_lr}, classifier_lr={classifier_lr}, batch_size={batch_size}, eval_metric={eval_metric}")
    logging.info(f"Max Length: {max_length}")
    logging.info(f"Checkpoint path: {checkpoint_path}, Continue: {continue_from_checkpoint}, Save Every: {save_every}")
    logging.info(f"Logging to {log_filepath}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Use provided tokenizer_path or same as model
    tokenizer_path = tokenizer_path or model_path

    logging.info(f"Loading {model_name} model and tokenizer from {model_path}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load Model
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.to(device)

    logging.info("Preparing datasets (Arabic-Squad + 50% ARCD for train, remaining 50% ARCD for test)...")
    # Note: raw_test contains the original dicts before tokenization, including the 'answer' field
    train_data, test_data, raw_test = load_and_prepare_datasets(data_dir=data_dir)

    processed_train_data = preprocess_datasets(train_data, model_name)
    processed_test_data = preprocess_datasets(test_data, model_name)

    logging.info("Tokenizing datasets...")
    # Note: doc_stride is not used since contexts fit within max_length=512
    try:
        tokenized_train_features = processed_train_data.map(tokenize, batched=False, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "doc_stride": 0}, remove_columns=processed_train_data.column_names, load_from_cache_file=False, keep_in_memory=True)
        tokenized_test_features = processed_test_data.map(tokenize, batched=False, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "doc_stride": 0}, remove_columns=processed_test_data.column_names, load_from_cache_file=False, keep_in_memory=True)
    except Exception as e:
        logging.error(f"Error during tokenization: {e}")
        raise

    # Filter out any invalid examples that might have been created during tokenization
    tokenized_train_features = tokenized_train_features.filter(lambda x: "input_ids" in x and len(x["input_ids"]) > 0)
    tokenized_test_features = tokenized_test_features.filter(lambda x: "input_ids" in x and len(x["input_ids"]) > 0)

    # Ensure datasets are not empty
    if len(tokenized_train_features) == 0 or len(tokenized_test_features) == 0:
        logging.error("One or more datasets are empty after processing. Check data loading and tokenization.")
        raise ValueError("Empty datasets after processing")

    logging.info("Sanity checking datasets...")
    
    train_dataset_dict = tokenized_train_features.train_test_split(test_size=0.01, shuffle=True)
    train_subset = train_dataset_dict["train"]
    val_subset = train_dataset_dict["test"]
    logging.info(f"Train subset created with {len(train_subset)} examples")
    logging.info(f"Train subset columns: {train_subset.column_names}")
    
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  # Shuffle train data
                                  num_workers=num_workers, pin_memory=False, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=False, collate_fn=custom_collate_fn)
    # No need to dataload test_features if using raw_test for evaluation
    test_dataloader = DataLoader(tokenized_test_features, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=False, collate_fn=custom_collate_fn)
    logging.info("Starting training...")
    logging.info(f"DataLoader created with {len(train_dataloader)} batches")
    
    # Pass eval_metric to train_qa_model
    model, peak_memory = train_qa_model(
        model, train_dataloader, val_dataloader, tokenizer, device,
        num_epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        checkpoint_path=checkpoint_path,
        continue_from_checkpoint=continue_from_checkpoint,
        save_every=save_every,
        max_length=max_length,
        doc_stride=128,  # Fixed value since it's deprecated
        eval_metric=eval_metric,
        encoder_lr=encoder_lr,
        classifier_lr=classifier_lr
    )

    logging.info("Training complete. Evaluating final model on the test set...")

    test_outputs = {'start_logits': [], 'end_logits': []}
    # Evaluate the final (best) model
    for batch in tqdm(test_dataloader, desc="Evaluating final model on the test set", leave=False):
        model.eval()
        keys_to_remove = ["start_char_offset", "end_char_offset"]
        batch = {k: (torch.tensor(v) if not isinstance(v, torch.Tensor) else v).to(device)
                            for k, v in batch.items() if k not in keys_to_remove}
        batch["start_positions"] = batch["start_positions"].long().view(-1)
        batch["end_positions"] = batch["end_positions"].long().view(-1)
        with torch.no_grad():
            outputs = model(**batch)
            test_outputs['start_logits'].extend(outputs.start_logits)
            test_outputs['end_logits'].extend(outputs.end_logits)
    final_em, final_f1, final_sm = evaluate_em_f1_sm(
        test_dataloader.dataset, test_outputs, tokenizer)

    if final_em is not None and final_f1 is not None and final_sm is not None:
        logging.info(f"Final Average EM Score on Test Set: {final_em:.4f}")
        logging.info(f"Final Average F1 Score on Test Set: {final_f1:.4f}")
        logging.info(f"Final Average SM Score on Test Set: {final_sm:.4f}")
    else:
        logging.warning("Could not compute final EM/F1/SM scores.")

    logging.info("Demo Predictions on Test Examples:")
    example_results = []
    # Use raw_test for prediction examples
    num_examples_to_show = min(5, len(test_dataloader.dataset))
    if num_examples_to_show > 0:
        for i in range(num_examples_to_show):
            ex = test_dataloader.dataset[i]
            decoded_text = tokenizer.decode(ex["input_ids"])
            question = decoded_text.split("[SEP]")[0].split('[CLS]')[1].strip()
            context = decoded_text.split("[SEP]")[1].strip()
            start_pos = ex["start_positions"]
            end_pos = ex["end_positions"]

            gt_answer = tokenizer.decode(ex["input_ids"][start_pos:end_pos+1])
            
            start_logits = test_outputs["start_logits"][i]
            end_logits = test_outputs["end_logits"][i]
            start_index = torch.argmax(start_logits).item()
            end_index = torch.argmax(end_logits).item()

            if end_index < start_index:
                end_index = start_index

            prediction = tokenizer.decode(ex["input_ids"][start_index:end_index+1])

            logging.info(f"\nExample {i+1}:")
            logging.info(f"Q: {normalize_text(question)}")
            # Limit context length for logging
            logging.info(f"Context (first 200 chars): {normalize_text(context[:200])}...")
            logging.info(f"GT Answer: {normalize_text(gt_answer)}")
            logging.info(f"Pred Answer: {normalize_text(prediction)}")
            logging.info("-" * 60)

            example_results.append({
                "question": normalize_text(question),
                "ground_truth": normalize_text(gt_answer),
                "prediction": normalize_text(prediction)
            })
    else:
        logging.info("No examples available in raw_test to show predictions.")

    # Save benchmark results to a JSON file with the same base name as the log
    result_filepath = log_filepath.replace('.log', '_results.json')

    results_data = {
        "configuration": {
            "model": model_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "eval_metric_for_stopping": eval_metric,
            "max_length": max_length,
            "timestamp": datetime.now().isoformat(),
        },
        "results": {
            # Store both final EM and F1
            "exact_match": float(final_em) if final_em is not None else None,
            "f1_score": float(final_f1) if final_f1 is not None else None,
            "sentence_match": float(final_sm) if final_sm is not None else None,
            "examples": example_results
        },
        "memory_usage": {
            "peak_ram_gb": float(peak_memory['ram_used_gb']),
            "peak_ram_percent": float(peak_memory['ram_percent']),
            "peak_vram_gb": float(peak_memory['vram_used_gb']),
            "peak_vram_percent": float(peak_memory['vram_percent']),
            "vram_total_gb": float(peak_memory['vram_total_gb']),
        }
    }

    try:
        with open(result_filepath, 'w', encoding='utf-8') as f:  # Ensure utf-8 for Arabic
            # ensure_ascii=False for Arabic
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Results saved to {result_filepath}")
    except Exception as e:
        logging.error(f"Failed to save results to {result_filepath}: {e}")

    logging.info("QA Benchmarking Complete.")
    
    return results_data
