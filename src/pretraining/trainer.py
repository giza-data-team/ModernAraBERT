"""
Training Module for ModernAraBERT

This module handles the MLM (Masked Language Modeling) pretraining of ModernAraBERT:
- LazyIterableTextDataset for memory-efficient data loading
- Distributed training with Accelerate
- Mixed precision (FP16) training
- Gradient accumulation and checkpointing
- Cosine learning rate scheduling with warmup
- Multi-stage sequence length training (128→512 tokens)
- Memory profiling and optimization
- Torch compile support for faster execution

Original file: "ModernBERT Training.py"
Status: Logic COMPLETELY UNCHANGED - only reorganized for modularity
"""

import os
import gc
import math
import shutil
import logging
import psutil
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Optional
from transformers import (
    AutoModelForMaskedLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_cosine_schedule_with_warmup,
    DataCollatorForLanguageModeling,
    pipeline
)
from accelerate import Accelerator
from accelerate.utils import set_seed

def setup_performance_optimizations(logger: logging.Logger):
    """
    Configure PyTorch and CUDA performance optimizations.

    Args:
        logger: Logger instance for logging configuration
    """
    # Performance optimizations for CUDA
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Enable cuDNN v8 API
    os.environ["TORCH_COMPILE_DEBUG"] = "0"  # Disable compilation debugging
    
    # Clear cache before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    # Check for tensor cores and configure accordingly
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap[0] >= 7:
            # Enable TF32 for Ampere+ GPUs (substantially faster with minimal precision loss)
            torch.set_float32_matmul_precision('high')
            logger.info("TensorFloat32 activated for faster matrix multiplications")


def log_memory_usage(logger: logging.Logger):
    """
    Log current RAM and GPU memory usage.

    Args:
        logger: Logger instance for output
    """
    proc = psutil.Process(os.getpid())
    ram_usage = proc.memory_info().rss / 1024**2
    gpu_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    logger.info(f"RAM usage: {ram_usage:.2f} MB, GPU usage: {gpu_usage:.2f} MB")
    
    # Detailed GPU memory info per device
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i} - Allocated: {torch.cuda.memory_allocated(i)/1e9:.2f} GB, "
                      f"Reserved: {torch.cuda.memory_reserved(i)/1e9:.2f} GB")


def detailed_memory_profile(model: PreTrainedModel, optimizer: Optional[torch.optim.Optimizer], logger: logging.Logger):
    """
    Log detailed memory usage including model parameters, largest layers, optimizer state, and CUDA statistics.

    Args:
        model: The model whose memory usage is profiled.
        optimizer: (Optional) Optimizer to include in memory profiling.
        logger: Logger instance for output
    """
    try:
        # Model parameters memory
        total_params_memory = sum(p.element_size() * p.nelement() for p in model.parameters())
        logger.info(f"Model parameters: {total_params_memory/1e9:.2f} GB")
        
        # Largest layers - identify potential memory bottlenecks
        big_layers = []
        for name, param in model.named_parameters():
            param_size = param.element_size() * param.nelement()
            if param_size > 50*1024*1024:  # Layers using >50MB
                big_layers.append((name, param_size/1e9))
        
        # Only log the top 3 largest layers to reduce log spam
        big_layers.sort(key=lambda x: x[1], reverse=True)
        for name, size in big_layers[:3]:
            logger.info(f"Large layer: {name}: {size:.4f} GB")
        
        # Optimizer states memory (if provided)
        if optimizer:
            optimizer_memory = 0
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        optimizer_memory += p.grad.element_size() * p.grad.nelement()
                    
                    # AdamW state (momentum and variance)
                    if p in optimizer.state:
                        for state_val in optimizer.state[p].values():
                            if torch.is_tensor(state_val):
                                optimizer_memory += state_val.element_size() * state_val.nelement()
            
            logger.info(f"Optimizer states: {optimizer_memory/1e9:.2f} GB")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            logger.info(f"Total CUDA memory allocated: {allocated/1e9:.2f} GB")
            logger.info(f"Total CUDA memory reserved: {reserved/1e9:.2f} GB")
            
            # Memory fragmentation
            if reserved > 0:
                fragmentation = 1.0 - (allocated / reserved)
                logger.info(f"Memory fragmentation: {fragmentation:.2%}")
                
                # Suggest action if fragmentation is high
                if fragmentation > 0.3 and allocated > 4e9:  # More than 30% fragmentation and using >4GB
                    logger.warning("High memory fragmentation detected. Consider increasing training batch size" 
                                 "or applying manual torch.cuda.empty_cache() if OOM issues occur.")
    except Exception as e:
        logger.error(f"Error in memory profiling: {e}")


class LazyIterableTextDataset(IterableDataset):
    """
    An iterable dataset that reads text files line-by-line from a directory,
    tokenizes them in batches, shuffles them using a buffer, and yields batches suitable for language modeling.

    Args:
        data_dir (str): Directory containing .txt files.
        tokenizer: Tokenizer to encode text.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        shuffle_buffer_size (int, optional): Buffer size for shuffling the text. Defaults to 10000.
        prefetch_factor (int, optional): Prefetch factor for data loading. Defaults to 5.
    """
    def __init__(self, data_dir, tokenizer, max_length=512, shuffle_buffer_size=10000, prefetch_factor=5):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
        self.shuffle_buffer_size = shuffle_buffer_size  # Buffer size for shuffle
        self.prefetch_factor = prefetch_factor  # Controls prefetching of data

    def __iter__(self):
        """
        Iterate over the dataset, reading lines from text files,
        shuffling with a buffer, batching, and tokenizing each batch.

        Yields:
            dict: A dictionary with tokenized 'input_ids', 'attention_mask', and 'labels'.
        """
        worker_info = get_worker_info()
        if worker_info is None:
            file_paths = self.file_paths
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            file_paths = self.file_paths[worker_id::num_workers]

        buffer = []
        
        def process_batch(text_batch):
            """
            Tokenize a batch of text and yield individual encoded examples.

            Args:
                text_batch (list of str): List of text lines to process.

            Yields:
                dict: Dictionary containing 'input_ids', 'attention_mask', and 'labels'.
            """
            encodings = self.tokenizer(
                text_batch,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            for i in range(len(text_batch)):
                input_ids = encodings["input_ids"][i]
                attention_mask = encodings["attention_mask"][i]
                labels = input_ids.clone()
                labels[input_ids == self.tokenizer.pad_token_id] = -100
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
        
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                        
                    # Add to buffer for shuffling
                    buffer.append(text)
                    if len(buffer) >= self.shuffle_buffer_size:
                        # Shuffle the buffer
                        import random
                        random.shuffle(buffer)
                        
                        # Process buffer in batches for efficiency
                        batch_size = 32  # Process multiple examples at once
                        for i in range(0, len(buffer), batch_size):
                            batch = buffer[i:i+batch_size]
                            yield from process_batch(batch)
                        
                        # Clear buffer
                        buffer = []
        
        # Process remaining items in buffer
        if buffer:
            import random
            random.shuffle(buffer)
            
            # Process remaining buffer in batches
            batch_size = 32
            for i in range(0, len(buffer), batch_size):
                batch = buffer[i:i+batch_size]
                yield from process_batch(batch)


def worker_init_fn(worker_id: int):
    """
    Initialize the random seed for data loader workers to ensure reproducibility.

    This function sets the seed for NumPy and Python's random module based on the worker's
    unique ID and PyTorch's initial seed, so that each worker produces deterministic random data.
    
    Args:
        worker_id (int): The ID of the data loader worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_path: str,
    accelerator: Accelerator,
    global_step: int,
    logger: logging.Logger
):
    """
    Save a checkpoint including model, optimizer, scheduler states, and current training step.

    Args:
        model: The model to save.
        tokenizer: Tokenizer associated with the model.
        optimizer: Optimizer whose state to save.
        scheduler: Learning rate scheduler state.
        checkpoint_path (str): Path where the checkpoint will be saved.
        accelerator: Accelerator object for handling distributed training.
        global_step (int): Current training step.
        logger: Logger instance for output
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f"Saving checkpoint at step {global_step} to {checkpoint_path}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.config.save_pretrained(checkpoint_path)
        accelerator.save_model(unwrapped_model, checkpoint_path)
        
        torch.save({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'global_step': global_step,
        }, os.path.join(checkpoint_path, "training_state.pt"))
        
        logger.info(f"Checkpoint successfully saved at: {checkpoint_path}")


def evaluate_model(
    model: PreTrainedModel,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
    fp16: bool,
    logger: logging.Logger
) -> Tuple[float, float]:
    """
    Evaluate the model on a validation set and compute loss and perplexity.

    Args:
        model: The model to evaluate.
        val_dataloader: DataLoader for the validation dataset.
        accelerator: Accelerator object for distributed evaluation.
        fp16: Whether to use mixed precision.
        logger: Logger instance for output

    Returns:
        tuple: Average loss and perplexity.
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(val_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=fp16):
        for batch in progress_bar:
            outputs = model(**batch)
            loss = outputs.loss
            loss = accelerator.gather(loss).mean()
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 100:
                break
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    
    model.train()
    return avg_loss, perplexity


def train_model(
    tokenizer: PreTrainedTokenizer,
    train_dir: str,
    val_dir: str,
    model_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 5e-7,
    max_length: int = 512,
    grad_acc_steps: int = 2,
    num_workers: int = 4,
    save_checkpoint_steps: int = 10000,
    warmup_ratio: float = 0.001,
    last_step: int = 0,
    fp16: bool = True,
    logging_steps: int = 100,
    use_torch_compile: bool = True,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Set up distributed training, prepare data loaders, and train the model.
    
    The training loop includes optimizer steps, learning rate scheduling,
    gradient accumulation, periodic checkpointing, and evaluation.
    
    Args:
        tokenizer: Tokenizer used for data processing.
        train_dir: Directory containing training text files
        val_dir: Directory containing validation text files
        model_path: Path to pretrained model to start from
        output_dir: Directory to save checkpoints and final model
        epochs: Number of training epochs (default: 3)
        batch_size: Per-GPU batch size (default: 32)
        learning_rate: Learning rate (default: 5e-7)
        max_length: Maximum sequence length (default: 512)
        grad_acc_steps: Gradient accumulation steps (default: 2)
        num_workers: Number of data loader workers (default: 4)
        save_checkpoint_steps: Save checkpoint every N steps (default: 10000)
        warmup_ratio: Warmup ratio for learning rate (default: 0.001)
        last_step: Last completed step for resuming (default: 0)
        fp16: Enable mixed precision training (default: True)
        logging_steps: Log metrics every N steps (default: 100)
        use_torch_compile: Enable torch.compile (default: True)
        seed: Random seed (default: 42)
        logger: Logger instance (optional, creates new if None)
    
    Returns:
        str: Output directory path where the final model is saved.
    """    
    # Set seed for reproducibility
    set_seed(seed)
    
    logger.info("Setting up accelerator for distributed training...")
    accelerator = Accelerator(
        mixed_precision="fp16" if fp16 else "no",
        gradient_accumulation_steps=grad_acc_steps,
        device_placement=True,
    )
    
    logger.info(f"Distributed training setup: {accelerator.state}")
    logger.info(f"Process count: {accelerator.num_processes}")
    logger.info(f"Using device: {accelerator.device}")
    
    effective_batch_size = batch_size * accelerator.num_processes * grad_acc_steps
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    adjusted_logging_steps = max(logging_steps // accelerator.num_processes, 1)
    logger.info("Loading training and validation data from lazy iterable raw text files...")
    
    train_dataset = LazyIterableTextDataset(train_dir, tokenizer, max_length=max_length)
    val_dataset = LazyIterableTextDataset(val_dir, tokenizer, max_length=max_length)
    
    # Configure data loading with optimized settings
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Dataset handles shuffling
        pin_memory=True, 
        num_workers=num_workers,
        collate_fn=data_collator,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,  # Increase prefetching for better throughput
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,  
        shuffle=False,
        pin_memory=True, 
        num_workers=max(1, num_workers // 2),  
        collate_fn=data_collator,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    logger.info("Loading model for training...")
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    
    # Apply gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    model.config.use_cache = False  # Disable KV caching for training
    
    # Configure optimizer with weight decay normalization and memory optimization
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,  
    )
    
    # Print a detailed Memory Profile for model and optimizer
    logger.info("Detailed memory profile BEFORE training:")
    detailed_memory_profile(model, optimizer, logger)

    # Calculate number of steps for scheduler
    # Using approximate number of samples for more accurate scheduling
    approx_num_samples = 6e6 
    num_update_steps_per_epoch = approx_num_samples // effective_batch_size
    max_train_steps = epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_ratio * max_train_steps)
    
    logger.info(f"Training steps per epoch: ~{num_update_steps_per_epoch}")
    logger.info(f"Total training steps: ~{max_train_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=max_train_steps
    )
    
    # Prepare model, optimizer and dataloaders with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    # Apply torch.compile for faster execution if enabled and available
    if use_torch_compile and hasattr(torch, "compile") and torch.__version__ >= "2.0.0":
        logger.info("Applying torch.compile() to model for accelerated execution")
        try:
            if torch.cuda.is_available():
                device_cap = torch.cuda.get_device_capability()
                mode = "reduce-overhead" if device_cap[0] < 7 else "max-autotune"
            else:
                mode = "reduce-overhead"
                
            # Apply compilation with optimizations
            model = torch.compile(
                model,
                mode=mode,
                fullgraph=False,  # Partial graph compilation is more stable
                dynamic=False,    # Static shapes for better optimization
                backend="inductor"  # Best backend for most models
            )
            logger.info(f"Model successfully compiled with mode '{mode}'")
        except Exception as e:
            logger.warning(f"Failed to apply torch.compile: {e}. Continuing with standard model.")

    # Add profiling after the model is moved to GPU and optimizer states are initialized
    logger.info("Detailed memory profile AFTER accelerator preparation:")
    detailed_memory_profile(model, optimizer, logger)
    
    # Resume from checkpoint if specified
    global_step = last_step
    steps_to_skip = last_step
    
    if last_step > 0 and os.path.exists(os.path.join(output_dir, f"checkpoint_step_{last_step}")):
        checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{last_step}")
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        
        if os.path.exists(training_state_path):
            logger.info(f"Loading optimizer and scheduler states from {training_state_path}")
            training_state = torch.load(training_state_path, map_location='cpu')
            
            # Load optimizer and scheduler states
            optimizer.load_state_dict(training_state['optimizer'])
            scheduler.load_state_dict(training_state['scheduler'])
            global_step = training_state.get('global_step', last_step)
            
            logger.info(f"Resumed from step {global_step}")
            
            # Log detailed memory profile after loading checkpoint
            logger.info("========================")
            logger.info("Detailed memory profile AFTER loading model and optimizer:")
            detailed_memory_profile(model, optimizer, logger)
            logger.info("========================")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    # Track throughput
    training_start_time = datetime.now()
    samples_processed = 0
    
    for epoch in range(epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        step = 0
        epoch_start_time = datetime.now()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            # Skip until we reach the last checkpoint step in first epoch
            if epoch == 0 and step < steps_to_skip:
                step += 1
                progress_bar.update(1)
                continue
                
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                # Update steps only when we do optimizer step
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  
                    
                    global_step += 1
                    samples_processed += effective_batch_size
                    
                    # Calculate and log throughput periodically
                    if global_step % 100 == 0 and step > 100:
                        elapsed_time = (datetime.now() - training_start_time).total_seconds()
                        samples_per_second = samples_processed / elapsed_time
                        logger.info(f"Throughput: {samples_per_second:.2f} samples/second")
                    
                    # Reduced frequency memory profiling
                    if global_step % 1000 == 0:
                        logger.info("========================")
                        logger.info(f"MEMORY PROFILE AT STEP {global_step}")
                        detailed_memory_profile(model, optimizer, logger)
                        logger.info("========================")
                    
                    # Log training progress
                    if global_step % adjusted_logging_steps == 0:
                        # Gather loss from all processes
                        gathered_loss = accelerator.gather(loss).mean().item()
                        current_lr = scheduler.get_last_lr()[0]
                        
                        if accelerator.is_local_main_process:
                            samples_per_second = samples_processed / ((datetime.now() - training_start_time).total_seconds()) if step > 100 else 0
                            progress_bar.set_postfix({
                                'loss': f'{gathered_loss:.4f}',
                                'lr': f'{current_lr:.6f}',
                                'step': global_step,
                                'samples/s': f'{samples_per_second:.1f}' if step > 100 else 'N/A'
                            })
                            
                            logger.info(f"Epoch {epoch + 1}, Step {global_step}: "
                                    f"Loss {gathered_loss:.4f}, LR {current_lr:.6f}")
                            log_memory_usage(logger)
                    
                    # Checkpointing logic with less frequent evaluations
                    if global_step % save_checkpoint_steps == 0:
                        logger.info(f"Evaluating and checkpointing at step {global_step}...")
                        
                        # Evaluate on validation set
                        val_loss, perplexity = evaluate_model(model, val_dataloader, accelerator, fp16, logger)
                        
                        if accelerator.is_local_main_process:
                            logger.info(f"Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")
                        
                        # Save checkpoint
                        checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{global_step}")
                        save_checkpoint(model, tokenizer, optimizer, scheduler, checkpoint_path, accelerator, global_step, logger)
                        
                        # Save best model separately
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_path = os.path.join(output_dir, "best_model")
                            
                            if accelerator.is_local_main_process:
                                logger.info(f"New best model with validation loss: {val_loss:.4f}")
                                # Just create a symlink to save disk space
                                if os.path.exists(best_model_path):
                                    if os.path.islink(best_model_path):
                                        os.unlink(best_model_path)
                                    else:
                                        shutil.rmtree(best_model_path)
                                os.symlink(checkpoint_path, best_model_path)
                
                # Accumulate loss for epoch average (use detached loss)
                total_loss += loss.detach().float()
                step += 1
                
                # Less frequent cache clearing to improve throughput - only clear every 100 steps
                if step % 100 == 0:
                    accelerator.clear()
                
        # Reset steps to skip after first epoch
        steps_to_skip = 0
        
        # Calculate and log epoch stats
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        avg_train_loss = accelerator.gather(total_loss).mean().item() / step if step > 0 else 0
        if accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. Average Training Loss: {avg_train_loss:.4f}")
        
        # Evaluate at end of epoch
        val_loss, perplexity = evaluate_model(model, val_dataloader, accelerator, fp16, logger)
        if accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")
    
    # Final evaluation and save
    logger.info("Training completed, running final evaluation...")
    val_loss, perplexity = evaluate_model(model, val_dataloader, accelerator, fp16, logger)
    
    total_training_time = (datetime.now() - training_start_time).total_seconds()
    logger.info(f"Total training time: {total_training_time:.2f} seconds")
    logger.info(f"Average throughput: {samples_processed/total_training_time:.2f} samples/second")
    
    if accelerator.is_local_main_process:
        logger.info(f"Final Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # Save final model
        final_output_dir = os.path.join(output_dir, "final_model")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_output_dir)
        logger.info(f"Final model and tokenizer saved to {final_output_dir}")
    
    return output_dir


def demo_fill_mask(model_path: str, tokenizer: PreTrainedTokenizer, logger: logging.Logger):
    """
    Run a simple fill-mask pipeline demonstration with the trained model.
    
    Loads a masked language model, runs a demo input, and prints the predictions.

    Args:
        model_path (str): Path to the trained model.
        tokenizer: Tokenizer for the model.
        logger: Logger instance for output
    """
    print("\nRunning fill-mask pipeline demo:")
    
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        
        input_text = "اللغة [MASK] لغة جميلة."
        results = fill_mask(input_text)
        
        print("Input:", input_text)
        for result in results:
            print(f"Prediction: {result['sequence']} (Score: {result['score']:.4f})")
    except Exception as e:
        print(f"Demo error: {e}")
