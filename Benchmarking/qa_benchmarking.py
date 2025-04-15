import os
import io
import requests
import zipfile
import logging
import random
import numpy as np
import torch
import json
import copy
import math
from datetime import datetime
import argparse
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm
from torch.optim.adamw import AdamW
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import ModernBertModel, ModernBertPreTrainedModel, ModernBertConfig
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead
from datasets import load_dataset, Dataset, concatenate_datasets, Value


def predict_answer(model, tokenizer, question, context, device, max_length=512, doc_stride=128):
    if not isinstance(question, str):
        question = ""
    if not isinstance(context, str):
        context = ""
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    offset_mapping = inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item()
    tokens = inputs["input_ids"][0][start_index:end_index+1]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def evaluate_bleu_rouge(model, raw_dataset, tokenizer, device, max_length=512, doc_stride=128, metric="bleu"):
    model.eval()
    total_score = 0
    count = 0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if metric == "rouge" else None

    with torch.no_grad():
        for ex in tqdm(raw_dataset, desc="Evaluating", leave=False):
            question = ex.get("question", "")
            context = ex.get("context", "")
            reference = ex.get("answer", "")

            prediction = predict_answer(model, tokenizer, question, context, device, max_length, doc_stride)
            if metric == "bleu":
                score = sentence_bleu([reference.split()], prediction.split())
            else:
                rouge_scores = scorer.score(reference, prediction)
                score = rouge_scores["rougeL"].fmeasure
            total_score += score
            count += 1

    return total_score / count if count > 0 else None


def train_qa_model(model, train_dataloader, val_raw_dataset, tokenizer, device,
                   num_epochs=3, learning_rate=2e-5, patience=2, checkpoint_path=None,
                   continue_from_checkpoint=False, save_every=None, max_length=512, doc_stride=128,
                   metric="bleu"):

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scaler = torch.amp.GradScaler(device=device)

    if continue_from_checkpoint and checkpoint_path is not None and os.path.exists(checkpoint_path):
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    
    best_val_score = float('-inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_index, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            optimizer.zero_grad()
            batch_no_labels = {k: v.to(device) for k, v in batch.items() if k not in ["start_positions", "end_positions"]}
            batch_labels = {
                "start_positions": batch["start_positions"].long().view(-1).to(device),
                "end_positions": batch["end_positions"].long().view(-1).to(device)
            }
            with torch.amp.autocast(device_type=device.type):
                outputs = model(**batch_no_labels)
                loss_start = torch.nn.functional.cross_entropy(outputs.start_logits, batch_labels["start_positions"])
                loss_end = torch.nn.functional.cross_entropy(outputs.end_logits, batch_labels["end_positions"])
                loss = (loss_start + loss_end) / 2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            logging.info(f"Epoch {epoch+1}, Batch {batch_index+1}: Loss = {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")

        avg_val_score = evaluate_bleu_rouge(model, val_raw_dataset, tokenizer, device, max_length, doc_stride, metric=metric)
        if avg_val_score is not None:
            logging.info(f"Validation {metric.upper()} Score after Epoch {epoch+1}: {avg_val_score:.4f}")
            print(f"Validation {metric.upper()} Score after Epoch {epoch+1}: {avg_val_score:.4f}")
        else:
            logging.info(f"Validation {metric.upper()} Score after Epoch {epoch+1}: N/A")
            print(f"Validation {metric.upper()} Score after Epoch {epoch+1}: N/A")

        if avg_val_score is not None and avg_val_score > best_val_score:
            best_val_score = avg_val_score
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logging.info("Validation score improved; resetting patience counter.")
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
        else:
            patience_counter += 1
            logging.info(f"No improvement in validation score. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                print("Early stopping triggered.")
                break

        if save_every is not None and checkpoint_path is not None and ((epoch+1) % save_every == 0):
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Periodic checkpoint saved at epoch {epoch+1} to {checkpoint_path}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


class ModernBertForQuestionAnswering(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2) 
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.post_init()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            sliding_window_mask: torch.Tensor = None,
            position_ids: torch.Tensor = None,
            start_positions: torch.Tensor = None,
            end_positions: torch.Tensor = None,
            indices: torch.Tensor = None,
            cu_seqlens: torch.Tensor = None,
            max_seqlen: int = None,
            batch_size: int = None,
            seq_len: int = None,
            output_attentions: bool = None,
            output_hidden_states: bool = None,
            return_dict: bool = None,
            **kwargs,
    ) -> QuestionAnsweringModelOutput:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Positions outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Positions outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs[0]
        sequence_output = self.drop(self.head(sequence_output))

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, add a dimension if necessary
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # Sometimes the start/end positions are outside our model inputs, so we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def get_log_filename(model_name, metric, epochs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"QA_Benchmark_{model_name}_{metric}_{epochs}ep_{timestamp}.log"


LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

DATA_DIRECTORY = "./Data/"
os.makedirs(DATA_DIRECTORY, exist_ok=True)


def download_and_extract(url, extract_to="."):
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


def flatten_squad_list(data_list):
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
                answers = qa.get("answers", [])
                if len(answers) > 0:
                    answer_text = answers[0]["text"]
                    start_char = answers[0]["answer_start"]
                else:
                    answer_text = ""
                    start_char = 0
                all_rows.append({
                    "context": context,
                    "question": question,
                    "answer": answer_text,
                    "answer_start": start_char
                })
    return all_rows


def load_and_flatten_squad(squad_path):
    raw = load_dataset("json", data_files={"train": squad_path})["train"]
    all_flat = []
    for row in raw:
        data_list = row["data"]
        sub_rows = flatten_squad_list(data_list)
        all_flat.extend(sub_rows)
    return Dataset.from_list(all_flat)


def fix_answer_fields(example):
    if isinstance(example["answer"], list):
        example["answer"] = example["answer"][0] if example["answer"] else ""
    if isinstance(example["answer_start"], list):
        example["answer_start"] = example["answer_start"][0] if example["answer_start"] else 0
    return example


def transform_arcd(example):
    if "answers" in example:
        if isinstance(example["answers"], list) and len(example["answers"]) > 0:
            first = example["answers"][0]
            example["answer"] = first.get("text", "")
            example["answer_start"] = first.get("answer_start", 0)
        elif isinstance(example["answers"], dict):
            example["answer"] = example["answers"].get("text", "")
            example["answer_start"] = example["answers"].get("answer_start", 0)
        else:
            example["answer"] = ""
            example["answer_start"] = 0
    else:
        example["answer"] = ""
        example["answer_start"] = 0
    for col in ["answers", "id", "title"]:
        if col in example:
            del example[col]
    return example


def squeeze_tokenized(tokenized):
    for key in tokenized:
        if isinstance(tokenized[key], torch.Tensor) and tokenized[key].shape[0] == 1:
            tokenized[key] = tokenized[key].squeeze(0)
    return tokenized


def prepare_qa_features(example, tokenizer, max_length, doc_stride):
    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    tokenized = squeeze_tokenized(tokenized)
    offsets = tokenized["offset_mapping"].tolist()
    tokenized.pop("offset_mapping")

    answer_start_char = example.get("answer_start", 0)
    answer_text = example.get("answer", "")
    answer_end_char = answer_start_char + len(answer_text)

    token_start_index = 0
    token_end_index = len(offsets) - 1
    for i, item in enumerate(offsets):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            start_val, end_val = item[:2]
            if start_val <= answer_start_char < end_val:
                token_start_index = i
                break
    for i, item in enumerate(offsets):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            start_val, end_val = item[:2]
            if start_val < answer_end_char <= end_val:
                token_end_index = i
                break

    tokenized["start_positions"] = torch.tensor(token_start_index)
    tokenized["end_positions"] = torch.tensor(token_end_index)
    return tokenized


def tokenize_for_eval(example, tokenizer, max_length, doc_stride):
    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    return squeeze_tokenized(tokenized)

def predict_answer(model, tokenizer, question, context, device, max_length=512, doc_stride=128):
    if not isinstance(question, str):
        question = ""
    if not isinstance(context, str):
        context = ""
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    offset_mapping = inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item()
    tokens = inputs["input_ids"][0][start_index:end_index+1]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def custom_collate_fn(batch):
    return torch.utils.data.dataloader.default_collate(batch)


def load_and_prepare_datasets(tokenizer, max_length, doc_stride):
    GITHUB_URL = "https://github.com/WissamAntoun/Arabic_QA_Datasets/raw/master/data.zip"
    DATA_DIR = "./data/data/data/data"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        download_and_extract(GITHUB_URL, extract_to=DATA_DIR)
    squad_path = os.path.join(DATA_DIR, "Arabic-SQuAD.json")
    if not os.path.exists(squad_path):
        raise FileNotFoundError(f"Arabic-Squad.json not found in {DATA_DIR}. Check extracted contents or rename accordingly.")
    raw_squad = load_dataset("json", data_files={"train": squad_path})["train"]
    all_flat = []
    for row in raw_squad:
        data_list = row["data"]
        sub_rows = flatten_squad_list(data_list)
        all_flat.extend(sub_rows)
    squad_flat = Dataset.from_list(all_flat)
    squad_flat = squad_flat.map(fix_answer_fields)
    squad_flat = squad_flat.cast_column("answer", Value("string"))
    squad_flat = squad_flat.cast_column("answer_start", Value("int64"))

    logging.info("Loading ARCD dataset from Hugging Face (hsseinmz/arcd)...")
    raw_arcd = load_dataset("hsseinmz/arcd")["train"]
    split_arcd = raw_arcd.train_test_split(test_size=0.5, seed=42)
    arcd_train = split_arcd["train"]
    arcd_test = split_arcd["test"]
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

    def tokenize_qa_for_train(ex):
        return prepare_qa_features(ex, tokenizer, max_length, doc_stride)

    def tokenize_qa_for_test(ex):
        return tokenize_for_eval(ex, tokenizer, max_length, doc_stride)

    logging.info("Tokenizing combined training set...")
    train_features = combined_train.map(
        tokenize_qa_for_train, batched=False, remove_columns=combined_train.column_names)
    train_features = train_features.filter(lambda x: "input_ids" in x)
    logging.info("Tokenizing test set (ARCD test split)...")
    test_features = arcd_test.map(tokenize_qa_for_test, batched=False)
    test_features = test_features.filter(lambda x: "input_ids" in x)
    logging.info(f"Final training set has {len(train_features)} examples.")
    logging.info(f"Final test set has {len(test_features)} examples.")

    train_features.set_format("torch")
    test_features.set_format("torch")

    return train_features, test_features, arcd_test


def main():
    """
    Arguments:
    - --model-name: Choose between 'arabert' or 'modernbert' models.
    - --max-length: Maximum token sequence length for inputs.
    - --doc-stride: Overlap size when splitting long documents.
    - --batch-size: Number of samples per batch.
    - --epochs: Number of training epochs.
    - --learning-rate: Optimizer learning rate.
    - --checkpoint: Path to save or load model checkpoint.
    - --continue-from-checkpoint: Resume training from a saved checkpoint.
    - --save-every: Save checkpoint every N epochs.
    - --metric: Evaluation metric to use: 'bleu' or 'rouge'.
    """

    parser = argparse.ArgumentParser("QA Model Benchmarking")
    parser.add_argument("--model-name", type=str,
                        default="arabert", choices=["arabert", "modernbert"])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--continue-from-checkpoint", type=bool, default=False)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--metric", type=str, default="bleu",
                        choices=["bleu", "rouge"])
    args = parser.parse_args()

    # Generate log filename and setup logging
    log_filename = get_log_filename(args.model_name, args.metric, args.epochs)
    log_filepath = os.path.join(LOG_DIR, log_filename)

    # Set up logging with both console and file handlers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_filepath)],
        force=True
    )

    logging.info(f"Starting QA benchmarking for model {args.model_name}")
    logging.info(
        f"Configuration: epochs={args.epochs}, learning_rate={args.learning_rate}, batch_size={args.batch_size}, metric={args.metric}")
    logging.info(f"Logging to {log_filepath}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model_paths = {
        "arabert": "aubmindlab/bert-base-arabert",
        "modernbert": "./model_checkpoints/checkpoint_step_13000/"
    }
    model_path = model_paths.get(args.model_name)

    logging.info(
        f"Loading {args.model_name} model and tokenizer from {model_path}")

    if args.model_name == "arabert":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "./Tokenizer")

    if args.model_name == "arabert":
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    else:
        model = ModernBertForQuestionAnswering.from_pretrained(model_path)
    model.to(device)

    logging.info(
        "Preparing datasets (Arabic-Squad + 50% ARCD for train, remaining 50% for test)...")
    train_features, test_features, raw_test = load_and_prepare_datasets(
        tokenizer, args.max_length, args.doc_stride)

    train_dataloader = DataLoader(train_features, batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_features, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)

    logging.info("Starting training...")
    model = train_qa_model(
        model, train_dataloader, raw_test, tokenizer, device,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=2,
        checkpoint_path=args.checkpoint,
        continue_from_checkpoint=args.continue_from_checkpoint,
        save_every=args.save_every,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        metric=args.metric
    )

    final_score = evaluate_bleu_rouge(
        model, raw_test, tokenizer, device, args.max_length, args.doc_stride, metric=args.metric)
    if final_score is not None:
        logging.info(
            f"Final Average {args.metric.upper()} Score on Validation Set: {final_score:.4f}")
    else:
        logging.warning("Could not compute final metric score.")

    logging.info("Demo Predictions on Test Examples:")
    example_results = []
    for i in range(min(5, len(raw_test))):
        ex = raw_test[i]
        question = ex.get("question", "")
        context = ex.get("context", "")
        ground_truth = ex.get("answer", "N/A")
        predicted = predict_answer(
            model, tokenizer, question, context, device, args.max_length, args.doc_stride)

        logging.info(f"\nExample {i+1}:")
        logging.info(f"Q: {question}")
        logging.info(f"Context (first 200 chars): {context[:200]}...")
        logging.info(f"GT Answer: {ground_truth}")
        logging.info(f"Pred Answer: {predicted}")
        logging.info("-" * 60)

        example_results.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": predicted
        })

    # Save benchmark results to a JSON file with the same base name as the log
    result_filepath = log_filepath.replace('.log', '_results.json')

    results_data = {
        "configuration": {
            "model": args.model_name,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "metric": args.metric,
            "max_length": args.max_length,
            "doc_stride": args.doc_stride
        },
        "results": {
            f"{args.metric}_score": float(final_score) if final_score is not None else None,
            "examples": example_results
        }
    }

    with open(result_filepath, 'w') as f:
        json.dump(results_data, f, indent=2)
    logging.info(f"Results saved to {result_filepath}")

    logging.info("QA Benchmarking Complete.")


if __name__ == "__main__":
    main()
