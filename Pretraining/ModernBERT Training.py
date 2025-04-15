import os
import gc
import math
import re
import json
import warnings
from collections import Counter
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from datetime import datetime
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    DataCollatorForLanguageModeling,
    pipeline
)
import logging
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset, Dataset, DatasetDict
from tokenizers.pre_tokenizers import Whitespace
import torch._dynamo as dynamo

# Set a seed for reproducibility
set_seed(42)

LOG_FILE = 'Training-Memroy-Investigation.log'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)]
)

logger = logging.getLogger(__name__)
logger.info("Logging initialized successfully!")

# Training configuration
TRAIN_DIR = "./train/"
VAL_DIR = "./validation/"
TEST_DIR = "./test/"
TOKENIZER_PATH = "./Tokenizer/"
MODEL_PATH = "./output/checkpoint_step_70000/"
EPOCHS = 3
BATCH_SIZE = 32  # Per GPU batch size
LEARNING_RATE = 5e-7
MAX_LENGTH = 512
GRAD_ACC_STEPS = 2
NUM_WORKER = 4
SAVE_CHECKPOINT_STEPS = 10000
WARMUP_RATIO = 0.001  # Increased warmup ratio for more stable multi-GPU training
OUTPUT_DIR = "./efficient_training_output/"
LAST_STEP = 20000
FP16 = True  # Enable mixed precision training
LOGGING_STEPS = 100  # Log training metrics every N steps
USE_TORCH_COMPILE = True  # Enable torch.compile for faster training

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

def log_memory_usage():
    proc = psutil.Process(os.getpid())
    ram_usage = proc.memory_info().rss / 1024**2
    gpu_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    logger.info(f"RAM usage: {ram_usage:.2f} MB, GPU usage: {gpu_usage:.2f} MB")
    
    # Detailed GPU memory info per device
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i} - Allocated: {torch.cuda.memory_allocated(i)/1e9:.2f} GB, "
                      f"Reserved: {torch.cuda.memory_reserved(i)/1e9:.2f} GB")

class LazyIterableTextDataset(IterableDataset):
    def __init__(self, data_dir, tokenizer, max_length=512, shuffle_buffer_size=10000, prefetch_factor=5):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
        self.shuffle_buffer_size = shuffle_buffer_size  # Buffer size for shuffle
        self.prefetch_factor = prefetch_factor  # Controls prefetching of data

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            file_paths = self.file_paths
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            file_paths = self.file_paths[worker_id::num_workers]

        buffer = []
        
        # Batch processing for efficiency
        def process_batch(text_batch):
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

def save_checkpoint(model, tokenizer, optimizer, scheduler, checkpoint_path, accelerator, global_step):
    """Save model checkpoint with additional training state"""
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Wait for all processes to sync
    accelerator.wait_for_everyone()
    
    # Only save on main process
    if accelerator.is_main_process:
        logger.info(f"Saving checkpoint at step {global_step} to {checkpoint_path}")
        
        # Save model config
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.config.save_pretrained(checkpoint_path)
        
        # Save model weights
        accelerator.save_model(unwrapped_model, checkpoint_path)
        
        # tokenizer.save_pretrained(checkpoint_path)
        
        # Save optimizer and scheduler states
        torch.save({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'global_step': global_step,
        }, os.path.join(checkpoint_path, "training_state.pt"))
        
        logger.info(f"Checkpoint successfully saved at: {checkpoint_path}")

def evaluate_model(model, val_dataloader, accelerator):
    """Evaluate model on validation set with optimized performance"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(val_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)
    
    # Using torch.cuda.amp.autocast for mixed precision during evaluation
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=FP16):
        for batch in progress_bar:
            outputs = model(**batch)
            loss = outputs.loss
            
            # Gather loss from all processes
            loss = accelerator.gather(loss).mean()
            total_loss += loss.item()
            num_batches += 1
            
            # Limit validation to a reasonable number of batches for speed
            if num_batches >= 100:
                break
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    
    model.train()
    return avg_loss, perplexity

def train_model(tokenizer):
    logger.info("Setting up accelerator for distributed training...")
    
    # Initialize accelerator with the right configurations
    accelerator = Accelerator(
        mixed_precision="fp16" if FP16 else "no",
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        device_placement=True,
    )
    
    logger.info(f"Distributed training setup: {accelerator.state}")
    logger.info(f"Process count: {accelerator.num_processes}")
    logger.info(f"Using device: {accelerator.device}")
    
    # Get effective batch size accounting for devices and gradient accumulation
    effective_batch_size = BATCH_SIZE * accelerator.num_processes * GRAD_ACC_STEPS
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    # Set logging interval based on distributed setup
    logging_steps = max(LOGGING_STEPS // accelerator.num_processes, 1)
    
    logger.info("Loading training and validation data from lazy iterable raw text files...")
    
    train_dataset = LazyIterableTextDataset(TRAIN_DIR, tokenizer, max_length=MAX_LENGTH)
    val_dataset = LazyIterableTextDataset(VAL_DIR, tokenizer, max_length=MAX_LENGTH)
    
    # Use a specific random seed per process for data loading
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        import numpy as np
        np.random.seed(worker_seed)
        import random
        random.seed(worker_seed)
    
    # Configure data loading with optimized settings
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  # Dataset handles shuffling
        pin_memory=True, 
        num_workers=NUM_WORKER,
        collate_fn=data_collator,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,  # Increase prefetching for better throughput
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE * 2,  
        shuffle=False,
        pin_memory=True, 
        num_workers=max(1, NUM_WORKER // 2),  
        collate_fn=data_collator,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    logger.info("Loading model for training...")
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
    
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
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,  
    )
    
    # Print a detailed Memory Profile for model and optimizer
    logger.info("Detailed memory profile BEFORE training:")
    detailed_memory_profile(model, optimizer)


    # Calculate number of steps for scheduler
    # Using approximate number of samples for more accurate scheduling
    approx_num_samples = 6e6 
    num_update_steps_per_epoch = approx_num_samples // effective_batch_size
    max_train_steps = EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = int(WARMUP_RATIO * max_train_steps)
    
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
    if USE_TORCH_COMPILE and hasattr(torch, "compile") and torch.__version__ >= "2.0.0":
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
    detailed_memory_profile(model, optimizer)
    
    # Resume from checkpoint if specified
    global_step = LAST_STEP
    steps_to_skip = LAST_STEP
    
    if LAST_STEP > 0 and os.path.exists(os.path.join(OUTPUT_DIR, f"checkpoint_step_{LAST_STEP}")):
        checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{LAST_STEP}")
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        
        if os.path.exists(training_state_path):
            logger.info(f"Loading optimizer and scheduler states from {training_state_path}")
            training_state = torch.load(training_state_path, map_location='cpu')
            
            # Load optimizer and scheduler states
            optimizer.load_state_dict(training_state['optimizer'])
            scheduler.load_state_dict(training_state['scheduler'])
            global_step = training_state.get('global_step', LAST_STEP)
            
            logger.info(f"Resumed from step {global_step}")
            
            # Log detailed memory profile after loading checkpoint
            logger.info("========================")
            logger.info("Detailed memory profile AFTER loading model and optimizer:")
            detailed_memory_profile(model, optimizer)
            logger.info("========================")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    # Track throughput
    training_start_time = datetime.now()
    samples_processed = 0
    
    for epoch in range(EPOCHS):
        logger.info(f"Starting Epoch {epoch + 1}/{EPOCHS}")
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
                        detailed_memory_profile(model, optimizer)
                        logger.info("========================")
                    
                    # Log training progress
                    if global_step % logging_steps == 0:
                        # Gather loss from all processes
                        gathered_loss = accelerator.gather(loss).mean().item()
                        current_lr = scheduler.get_last_lr()[0]
                        
                        if accelerator.is_local_main_process:
                            progress_bar.set_postfix({
                                'loss': f'{gathered_loss:.4f}',
                                'lr': f'{current_lr:.6f}',
                                'step': global_step,
                                'samples/s': f'{samples_per_second:.1f}' if step > 100 else 'N/A'
                            })
                            
                            logger.info(f"Epoch {epoch + 1}, Step {global_step}: "
                                    f"Loss {gathered_loss:.4f}, LR {current_lr:.6f}")
                            log_memory_usage()
                    
                    # Checkpointing logic with less frequent evaluations
                    if global_step % SAVE_CHECKPOINT_STEPS == 0:
                        logger.info(f"Evaluating and checkpointing at step {global_step}...")
                        
                        # Evaluate on validation set
                        val_loss, perplexity = evaluate_model(model, val_dataloader, accelerator)
                        
                        if accelerator.is_local_main_process:
                            logger.info(f"Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")
                        
                        # Save checkpoint
                        checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}")
                        save_checkpoint(model, tokenizer, optimizer, scheduler, checkpoint_path, accelerator, global_step)
                        
                        # Save best model separately
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_path = os.path.join(OUTPUT_DIR, "best_model")
                            
                            if accelerator.is_local_main_process:
                                logger.info(f"New best model with validation loss: {val_loss:.4f}")
                                # Just create a symlink to save disk space
                                if os.path.exists(best_model_path):
                                    if os.path.islink(best_model_path):
                                        os.unlink(best_model_path)
                                    else:
                                        import shutil
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
        val_loss, perplexity = evaluate_model(model, val_dataloader, accelerator)
        if accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")
    
    # Final evaluation and save
    logger.info("Training completed, running final evaluation...")
    val_loss, perplexity = evaluate_model(model, val_dataloader, accelerator)
    
    total_training_time = (datetime.now() - training_start_time).total_seconds()
    logger.info(f"Total training time: {total_training_time:.2f} seconds")
    logger.info(f"Average throughput: {samples_processed/total_training_time:.2f} samples/second")
    
    if accelerator.is_local_main_process:
        logger.info(f"Final Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # Save final model
        final_output_dir = os.path.join(OUTPUT_DIR, "final_model")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_output_dir)
        logger.info(f"Final model and tokenizer saved to {final_output_dir}")
    
    return OUTPUT_DIR

def detailed_memory_profile(model, optimizer=None):
    """Profile detailed memory usage during training with better categorization"""
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


def main():
    # Configure thread settings for better CPU utilization
    if torch.get_num_threads() > NUM_WORKER * 2:
        # Limit threads to avoid CPU oversubscription
        torch.set_num_threads(NUM_WORKER * 2)
        logger.info(f"Set PyTorch threads to {NUM_WORKER * 2} for better CPU efficiency")
    
    # Initial memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Log system info
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA devices")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"CUDA Device {i}: {props.name}, Compute: {props.major}.{props.minor}, "
                      f"Memory: {props.total_memory/1e9:.2f} GB")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    
    # Clear memory again after loading tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    # Start training
    trained_model_path = train_model(tokenizer)
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Run demo on main process only
    if Accelerator().is_local_main_process:
        demo_fill_mask(trained_model_path, tokenizer)

def demo_fill_mask(model_path, tokenizer):
    """Run a simple demo of the trained model"""
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

if __name__ == "__main__":
    main()
