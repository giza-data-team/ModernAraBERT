import torch
from torch.optim.adamw import AdamW
import logging
import copy
import math
from sklearn.metrics import accuracy_score, classification_report
import os
from tqdm import tqdm  

def train_model(model, train_dataloader, val_dataloader=None, device=None, num_epochs=3, learning_rate=2e-5,
                patience=2, checkpoint_path=None, continue_from_checkpoint=False, save_every=None):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if continue_from_checkpoint and checkpoint_path is not None and os.path.exists(checkpoint_path):
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            log_loss = loss.item
            logging.info(f"Training Epoch {epoch+1} Batch Loss = {loss.item():.4f}")
            
        avg_train_loss = epoch_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")

        if val_dataloader:
            _, _, _, _, _, avg_val_loss, _ = evaluate_model(model, val_dataloader, device)
            logging.info(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")
            print(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                logging.info("Validation loss improved; resetting patience counter.")
                if checkpoint_path is not None:
                    torch.save(model.state_dict(), checkpoint_path)
                    logging.info(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
            else:
                patience_counter += 1
                logging.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    print("Early stopping triggered.")
                    break
        else:
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                logging.info("Train loss improved; resetting patience counter.")
                if checkpoint_path is not None:
                    torch.save(model.state_dict(), checkpoint_path)
                    logging.info(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
            else:
                patience_counter += 1
                logging.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    print("Early stopping triggered.")
                    break
        
        if save_every is not None and checkpoint_path is not None and ((epoch + 1) % save_every == 0):
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Periodic checkpoint saved at epoch {epoch+1} to {checkpoint_path}")

    # If validation wasn't used, still save final model state
    if not val_dataloader:
        best_model_state = copy.deepcopy(model.state_dict())
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
    

def evaluate_model(model, eval_dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    total_loss = 0
    total_batches = 0
    
    for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
        else:
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
        total_loss += loss.item()
        total_batches += 1
        probs = torch.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        all_confidences.extend(max_probs.cpu().tolist())
        labels_batch = batch["labels"].cpu().tolist()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels_batch)
        logging.info(f"Evaluation Batch Loss = {loss.item():.4f}")

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    perplexity = math.exp(avg_loss)
    accuracy = accuracy_score(all_labels, all_preds)
    
    if model.config.num_labels == 4:
         target_names = ["OBJ", "POS", "NEG", "NEU"]
         report = classification_report(all_labels, all_preds, labels=[0, 1, 2, 3],
                                        target_names=target_names, zero_division=0)
         macro_f1 = classification_report(all_labels, all_preds, labels=[0, 1, 2, 3], 
                                         output_dict=True, zero_division=0)['macro avg']['f1-score']
    elif model.config.num_labels == 2:
         report = classification_report(all_labels, all_preds, labels=[0, 1],
                                        target_names=["Negative", "Positive"], zero_division=0)
         macro_f1 = classification_report(all_labels, all_preds, labels=[0, 1], 
                                         output_dict=True, zero_division=0)['macro avg']['f1-score']
    else:
         report = classification_report(all_labels, all_preds, zero_division=0)
         macro_f1 = classification_report(all_labels, all_preds, 
                                         output_dict=True, zero_division=0)['macro avg']['f1-score']
    
    return macro_f1, report, all_preds, all_labels, all_confidences, avg_loss, perplexity
