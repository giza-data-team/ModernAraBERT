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
import psutil
import time
from collections import Counter
from datetime import datetime
import argparse
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
    """
    Predict the answer span from the context for the given question.

    Args:
        model: The QA model.
        tokenizer: The tokenizer associated with the model.
        question (str): The question to answer.
        context (str): The context paragraph.
        device: Torch device (cpu or cuda).
        max_length (int, optional): Maximum length of tokenized inputs. Defaults to 512.
        doc_stride (int, optional): Token overlap when splitting long documents. Defaults to 128.

    Returns:
        str: The decoded answer text.
    """
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
        # return_offsets_mapping=True,
        return_tensors="pt"
    )
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


# --- Start: New EM/F1 Evaluation Logic ---

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        # Simple removal for English 'a', 'an', 'the'. Adapt for Arabic 'ال' if needed.
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


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


def get_sentence_index(context, text):
    """Returns the index where the text falls in context if any, else returns -1"""
    sentences = re.split(r"(?<=[.؟!])\s+", context.strip())
    for idx, sent in enumerate(sentences):
        if text in sent:
            return idx
    return -1


def evaluate_em_f1_sm(model, raw_dataset, tokenizer, device, max_length=512, doc_stride=128):
    """
    Evaluate the QA model on a raw dataset and compute Exact Match (EM) and F1 score.
    Note: This implementation compares against a single ground truth answer per question
          due to the current data preprocessing. Standard SQuAD eval compares against all.

    Args:
        model: The QA model.
        raw_dataset (iterable): A dataset of examples containing "question", "context", and "answer".
        tokenizer: Tokenizer for preprocessing inputs.
        device: Torch device (cpu or cuda).
        max_length (int, optional): Maximum input length. Defaults to 512.
        doc_stride (int, optional): Document stride size. Defaults to 128.

    Returns:
        tuple (float, float) or (None, None): The average EM and F1 scores, or None if no examples processed.
    """
    model.eval()
    total_em = 0
    total_f1 = 0
    total_sm = 0
    count = 0

    with torch.no_grad():
        for ex in tqdm(raw_dataset, desc="Evaluating EM/F1/SM", leave=False):
            question = ex.get("question", "")
            context = ex.get("context", "")
            # Assumes 'answer' field holds the single ground truth string after preprocessing
            reference = ex.get("answer", "")

            # Skip if essential components are missing
            if not question or not context:
                logging.warning(
                    f"Skipping evaluation for example due to missing question/context: {ex}")
                continue

            prediction = predict_answer(
                model, tokenizer, question, context, device, max_length, doc_stride)

            # Calculate EM and F1 against the single reference
            em_score = compute_exact(prediction, reference)
            f1_score = compute_f1(prediction, reference)

            # Sentence Match
            pred_sent_idx = get_sentence_index(context, prediction)
            ref_sent_idx = get_sentence_index(context, reference)
            if pred_sent_idx != -1 and pred_sent_idx == ref_sent_idx:
                total_sm += 1

            total_em += em_score
            total_f1 += f1_score
            count += 1

    avg_em = total_em / count if count > 0 else None
    avg_f1 = total_f1 / count if count > 0 else None
    avg_sm = total_sm / count if count > 0 else None

    return avg_em, avg_f1, avg_sm


def train_qa_model(model, train_dataloader, val_raw_dataset, tokenizer, device,
                   num_epochs=3, learning_rate=2e-5, patience=10, checkpoint_path=None,
                   continue_from_checkpoint=False, save_every=None, max_length=512, doc_stride=128,
                   eval_metric="f1"):  # Changed metric to eval_metric, default 'f1'
    """
    Train the QA model over several epochs and evaluate on a validation set using EM/F1.

    Implements gradient scaling for mixed precision, early stopping based on F1 score,
    and checkpointing of the model state when the validation F1 improves.

    Args:
        model: The QA model to train.
        train_dataloader: DataLoader for training samples.
        val_raw_dataset: Raw dataset (e.g., list of dicts) for validation.
        tokenizer: Tokenizer for decoding predictions.
        device: Torch device (cpu or cuda).
        num_epochs (int, optional): Number of training epochs. Defaults to 3.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 2e-5.
        patience (int, optional): Number of epochs to wait for F1 improvement before stopping. Defaults to 2.
        checkpoint_path (str, optional): Path to save model checkpoints. Defaults to None.
        continue_from_checkpoint (bool, optional): Whether to resume from a checkpoint. Defaults to False.
        save_every (int, optional): Save a checkpoint every N epochs. Defaults to None.
        max_length (int, optional): Maximum length for tokenization. Defaults to 512.
        doc_stride (int, optional): Document stride length. Defaults to 128.
        eval_metric (str, optional): Primary metric for early stopping ('f1' or 'em'). Defaults to "f1".

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
    
    optimizer = AdamW(filter(lambda p: p.requires_grad,
                      model.parameters()), lr=learning_rate)
    scaler = torch.amp.GradScaler(device=device)

    if continue_from_checkpoint and checkpoint_path is not None and os.path.exists(checkpoint_path):
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)

    # Based on the primary eval_metric (F1 usually)
    best_val_score = float('-inf')
    patience_counter = 0
    best_model_state = None
    peak_memory = initial_memory.copy()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_index, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            optimizer.zero_grad()
            batch_no_labels = {k: v.to(device) for k, v in batch.items() if k not in [
                "start_positions", "end_positions"]}
            batch_labels = {
                "start_positions": batch["start_positions"].long().view(-1).to(device),
                "end_positions": batch["end_positions"].long().view(-1).to(device)
            }
            with torch.amp.autocast(device_type=device.type):
                outputs = model(**batch_no_labels)
                loss_start = torch.nn.functional.cross_entropy(
                    outputs.start_logits, batch_labels["start_positions"])
                loss_end = torch.nn.functional.cross_entropy(
                    outputs.end_logits, batch_labels["end_positions"])
                loss = (loss_start + loss_end) / 2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track peak memory usage during training
            current_memory = get_memory_usage()
            for key in current_memory:
                if key in peak_memory and current_memory[key] > peak_memory[key]:
                    peak_memory[key] = current_memory[key]
            epoch_loss += loss.item()
            logging.info(
                f"Epoch {epoch+1}, Batch {batch_index+1}: Loss = {loss.item():.4f}")

        avg_train_loss = epoch_loss / \
            len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        logging.info(
            f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")

        # Evaluate using EM/F1
        avg_em, avg_f1, avg_sm = evaluate_em_f1_sm(
            model, val_raw_dataset, tokenizer, device, max_length, doc_stride)

        if avg_em is not None and avg_f1 is not None and avg_sm is not None:
            logging.info(
                f"Validation EM Score after Epoch {epoch+1}: {avg_em:.4f}")
            logging.info(
                f"Validation F1 Score after Epoch {epoch+1}: {avg_f1:.4f}")
            logging.info(
                f"Validation SM Score after Epoch {epoch+1}: {avg_sm:.4f}")

            # Use the specified eval_metric for early stopping and checkpointing
            if eval_metric == "f1":
                current_score = avg_f1
            elif eval_metric == "em":
                current_score = avg_em
            else:
                current_score = avg_sm
            if current_score > best_val_score:
                best_val_score = current_score
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                logging.info(
                    f"Validation {eval_metric.upper()} score improved to {best_val_score:.4f}; resetting patience counter.")
                if checkpoint_path is not None:
                    torch.save(model.state_dict(), checkpoint_path)
                    logging.info(
                        f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
            else:
                patience_counter += 1
                logging.info(
                    f"No improvement in validation {eval_metric.upper()} score. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    break
        else:
            logging.warning(
                f"Validation EM/F1 Score after Epoch {epoch+1}: N/A (Evaluation failed or dataset empty)")
            # Decide how to handle this - maybe stop? Or continue? Currently continues.
            patience_counter += 1  # Count as no improvement
            if patience_counter >= patience:
                logging.info(
                    "Early stopping triggered due to evaluation failures or lack of improvement.")
                print(
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
            f"Loading best model state with {eval_metric.upper()} score: {best_val_score:.4f}")
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


class ModernBertForQuestionAnswering(ModernBertPreTrainedModel):
    """
    ModernBertForQuestionAnswering implements a question answering model based on ModernBERT.

    This class builds upon ModernBertPreTrainedModel by adding a prediction head
    for start and end positions and a dropout layer, and it computes the loss if labels are provided.
    """

    def __init__(self, config: ModernBertConfig):
        """
        Initialize ModernBertForQuestionAnswering.

        Args:
            config (ModernBertConfig): The model configuration.
        """
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
        """
        Forward pass of the model.

        Optionally computes the token classification loss for start and end positions if provided.

        Args:
            input_ids (torch.Tensor): Tensor of input token ids.
            attention_mask (torch.Tensor, optional): Mask tensor for inputs.
            sliding_window_mask (torch.Tensor, optional): Mask for sliding window.
            position_ids (torch.Tensor, optional): Positional ids tensor.
            start_positions (torch.Tensor, optional): Tensor of start positions for loss calculation.
            end_positions (torch.Tensor, optional): Tensor of end positions for loss calculation.
            indices, cu_seqlens, max_seqlen, batch_size, seq_len, output_attentions, output_hidden_states, return_dict: Additional parameters.
            **kwargs: Additional keyword arguments.

        Returns:
            QuestionAnsweringModelOutput: Model outputs including loss (if labels provided), start_logits, end_logits, hidden_states, and attentions.
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


# Modified get_log_filename
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


LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

DATA_DIRECTORY = "../Data/"
os.makedirs(DATA_DIRECTORY, exist_ok=True)


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


def load_and_flatten_squad(squad_path):
    """
    Load the SQuAD dataset from a JSON file, flatten its nested structure, and return a Dataset object.

    Args:
        squad_path (str): File path to the SQuAD JSON file.

    Returns:
        Dataset: A Hugging Face Dataset containing flattened QA examples.
    """
    raw = load_dataset("json", data_files={"train": squad_path})["train"]
    all_flat = []
    for row in raw:
        data_list = row["data"]
        sub_rows = flatten_squad_list(data_list)
        all_flat.extend(sub_rows)
    return Dataset.from_list(all_flat)


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


def prepare_qa_features(example, tokenizer, max_length, doc_stride):
    """
    Prepare tokenized features for a QA example for training.

    This function tokenizes the question and context, maps the character-level answer position
    to token positions, and returns a dictionary ready for training.

    Args:
        example (dict): A QA example containing 'question', 'context', 'answer', and 'answer_start'.
        tokenizer: The tokenizer to use.
        max_length (int): Maximum sequence length.
        doc_stride (int): Document stride for handling long contexts.

    Returns:
        dict: Tokenized features including input_ids, attention_mask, start_positions, and end_positions.
    """
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
    """
    Tokenize a QA example for evaluation without computing answer positions.

    Args:
        example (dict): A QA example containing 'question' and 'context'.
        tokenizer: The tokenizer to use.
        max_length (int): Maximum sequence length.
        doc_stride (int): Document stride.

    Returns:
        dict: Tokenized features for evaluation.
    """
    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    return squeeze_tokenized(tokenized)


def custom_collate_fn(batch):
    """
    Custom collate function that uses the default PyTorch collate.

    Args:
        batch (list): List of items from the dataset.

    Returns:
        A collated batch.
    """
    return torch.utils.data.dataloader.default_collate(batch)


def load_and_prepare_datasets(tokenizer, max_length, doc_stride):
    """
    Load and prepare the QA datasets by combining Arabic-SQuAD and ARCD,
    applying necessary transformations and tokenization.

    Args:
        tokenizer: The tokenizer to be used.
        max_length (int): Maximum sequence length.
        doc_stride (int): Document stride when tokenizing.

    Returns:
        tuple: (train_features, test_features, raw_test) where:
            - train_features: Tokenized training dataset.
            - test_features: Tokenized test dataset.
            - raw_test: Original test dataset examples.
    """
    GITHUB_URL = "https://github.com/WissamAntoun/Arabic_QA_Datasets/raw/master/data.zip"
    DATA_DIR = "../data/data/data/"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        download_and_extract(GITHUB_URL, extract_to=DATA_DIR)
    squad_path = os.path.join(DATA_DIR, "Arabic-SQuAD.json")
    if not os.path.exists(squad_path):
        raise FileNotFoundError(
            f"Arabic-Squad.json not found in {DATA_DIR}. Check extracted contents or rename accordingly.")
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
        """
        Tokenize a single QA example for training.

        This function applies the prepare_qa_features transformation using the given tokenizer,
        maximum sequence length, and document stride.

        Args:
            ex (dict): A QA example containing keys such as "question", "context", "answer", and "answer_start".

        Returns:
            dict: Tokenized example with keys for input_ids, attention_mask, start_positions, and end_positions.
        """
        return prepare_qa_features(ex, tokenizer, max_length, doc_stride)

    def tokenize_qa_for_test(ex):
        """
        Tokenize a single QA example for evaluation.

        This function applies tokenization for evaluation without computing answer positions,
        using the given tokenizer, maximum sequence length, and document stride.

        Args:
            ex (dict): A QA example containing keys such as "question" and "context".

        Returns:
            dict: Tokenized example suitable for evaluation.
        """
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
    Main entry point for QA model benchmarking using EM and F1 scores.

    Parses command-line arguments, sets up logging, loads the model and tokenizer,
    prepares the datasets, trains the QA model, evaluates its performance (EM/F1),
    and saves results.
    """
    """
    Command-line Arguments:
    - --model-name: Choose between 'arabert' or 'modernbert' models.
    - --max-length: Maximum token sequence length for inputs.
    - --doc-stride: Overlap size when splitting long documents.
    - --batch-size: Number of samples per batch.
    - --epochs: Number of training epochs.
    - --learning-rate: Optimizer learning rate.
    - --checkpoint: Path to save or load model checkpoint (best model based on F1).
    - --continue-from-checkpoint: Resume training from a saved checkpoint.
    - --save-every: Save checkpoint every N epochs (intermediate saves).
    - --eval-metric: Metric for early stopping ('f1' or 'em', default 'f1'). # Added clarification
    """

    parser = argparse.ArgumentParser(
        "QA Model Benchmarking (EM/F1)")  # Updated description
    parser.add_argument("--model-name", type=str,
                        default="arabert", choices=["arabert", "modernbert", "mbert"])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save the best model checkpoint (based on eval-metric).")
    parser.add_argument("--continue-from-checkpoint", action='store_true',  # Use action='store_true'
                        help="Resume training from the checkpoint specified by --checkpoint.")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Save an intermediate checkpoint every N epochs.")
    # Removed --metric, added --eval-metric
    parser.add_argument("--eval-metric", type=str, default="f1", choices=["f1", "em", "sm"],
                        help="Metric used for early stopping and selecting the best checkpoint.")
    args = parser.parse_args()

    # Generate log filename and setup logging (metric removed from filename)
    log_filename = get_log_filename(args.model_name, args.epochs)
    log_filepath = os.path.join(LOG_DIR, log_filename)

    # Set up logging with both console and file handlers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_filepath)],
        force=True  # Overwrite existing handlers if any
    )

    logging.info(
        f"Starting QA benchmarking for model {args.model_name} using EM/F1")
    logging.info(
        f"Configuration: epochs={args.epochs}, lr={args.learning_rate}, batch_size={args.batch_size}, eval_metric={args.eval_metric}")
    logging.info(
        f"Max Length: {args.max_length}, Doc Stride: {args.doc_stride}")
    logging.info(
        f"Checkpoint path: {args.checkpoint}, Continue: {args.continue_from_checkpoint}, Save Every: {args.save_every}")
    logging.info(f"Logging to {log_filepath}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model_paths = {
        "arabert": "aubmindlab/bert-base-arabert",
        # Ensure this path is correct for your setup
        "modernbert": "/gpfs/helios/home/abdelrah/ModernBERT/Training2/output/checkpoint_step_13000/",
        "mbert": "google-bert/bert-base-multilingual-cased"
    }
    model_path = model_paths.get(args.model_name)
    if not os.path.exists(model_path) and args.model_name == "modernbert":
        logging.warning(
            f"ModernBERT model path not found: {model_path}. Ensure the checkpoint exists.")
        # Decide if you want to exit or proceed (maybe download?)
        # return # Example: exit if path invalid

    logging.info(
        f"Loading {args.model_name} model and tokenizer from {model_path}")

    # Load Tokenizer - Assuming ModernBERT tokenizer is saved separately
    if args.model_name != "modernbert":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        # Ensure this path is correct for your setup
        tokenizer_path = "/gpfs/helios/home/abdelrah/ModernBERT/Training2/Tokenizer"
        if not os.path.exists(tokenizer_path):
            logging.error(
                f"ModernBERT tokenizer path not found: {tokenizer_path}. Cannot proceed.")
            return
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load Model
    if args.model_name == "arabert":
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    else:
        # Use the custom ModernBertForQuestionAnswering class
        model = ModernBertForQuestionAnswering.from_pretrained(model_path)
    model.to(device)

    logging.info(
        "Preparing datasets (Arabic-Squad + 50% ARCD for train, remaining 50% ARCD for test)...")
    # Note: raw_test contains the original dicts before tokenization, including the 'answer' field
    train_features, test_features, raw_test = load_and_prepare_datasets(
        tokenizer, args.max_length, args.doc_stride)

    # Ensure datasets are not empty
    if len(train_features) == 0 or len(test_features) == 0 or len(raw_test) == 0:
        logging.error(
            "One or more datasets are empty after processing. Check data loading and tokenization.")
        return

    train_dataloader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True,  # Shuffle train data
                                  num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)
    # No need to dataload test_features if using raw_test for evaluation
    # test_dataloader = DataLoader(test_features, batch_size=args.batch_size, shuffle=False,
    #                              num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)

    logging.info("Starting training...")
    # Pass eval_metric to train_qa_model
    model, peak_memory = train_qa_model(
        model, train_dataloader, raw_test, tokenizer, device,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=200,  # Or make this an arg
        checkpoint_path=args.checkpoint,
        continue_from_checkpoint=args.continue_from_checkpoint,
        save_every=args.save_every,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        eval_metric=args.eval_metric  # Pass the chosen metric
    )

    logging.info(
        "Training complete. Evaluating final model on the test set...")
    # Evaluate the final (best) model
    final_em, final_f1, final_sm = evaluate_em_f1_sm(
        model, raw_test, tokenizer, device, args.max_length, args.doc_stride)

    if final_em is not None and final_f1 is not None and final_sm is not None:
        logging.info(
            f"Final Average EM Score on Test Set: {final_em:.4f}")
        logging.info(
            f"Final Average F1 Score on Test Set: {final_f1:.4f}")
        logging.info(
            f"Final Average SM Score on Test Set: {final_sm:.4f}")
    else:
        logging.warning("Could not compute final EM/F1/SM scores.")

    logging.info("Demo Predictions on Test Examples:")
    example_results = []
    # Use raw_test for prediction examples
    num_examples_to_show = min(5, len(raw_test))
    if num_examples_to_show > 0:
        for i in range(num_examples_to_show):
            ex = raw_test[i]
            question = ex.get("question", "")
            context = ex.get("context", "")
            # Single ground truth from preprocessing
            ground_truth = ex.get("answer", "N/A")

            if not question or not context:
                logging.warning(
                    f"Skipping prediction example {i+1} due to missing question/context.")
                continue

            predicted = predict_answer(
                model, tokenizer, question, context, device, args.max_length, args.doc_stride)

            logging.info(f"\nExample {i+1}:")
            logging.info(f"Q: {question}")
            # Limit context length for logging
            logging.info(f"Context (first 200 chars): {context[:200]}...")
            logging.info(f"GT Answer: {ground_truth}")
            logging.info(f"Pred Answer: {predicted}")
            logging.info("-" * 60)

            example_results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": predicted
            })
    else:
        logging.info("No examples available in raw_test to show predictions.")

    # Save benchmark results to a JSON file with the same base name as the log
    result_filepath = log_filepath.replace('.log', '_results.json')

    results_data = {
        "configuration": {
            "model": args.model_name,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "eval_metric_for_stopping": args.eval_metric,  # Clarify role
            "max_length": args.max_length,
            "doc_stride": args.doc_stride,
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


if __name__ == "__main__":
    main()
