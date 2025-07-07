# trainer.py
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import utils
import os
import traceback

# Must match the value used in data_utils.py for padding labels
PAD_VALUE_LABELS = -100

def train_one_epoch(model, loader, criterion, optimizer, device, epoch_num, total_epochs):
    """Runs a single training epoch for the sequence model."""
    
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    if loader is None:
        print(f"\nWarning: Training loader for Epoch {epoch_num} is None. Skipping epoch.")
        return 0.0
    
    loader_pbar = tqdm(loader, desc=f"Train Epoch {epoch_num}/{total_epochs}", leave=False)

    num_classes = model.num_classes

    for features, targets, attention_mask in loader_pbar:
        features = features.to(device)
        targets = targets.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        try:
            final_logits = model(features, attention_mask)

            loss = criterion(final_logits.view(-1, num_classes), targets.view(-1))

            if torch.isnan(loss):
                print(f"\nWarning: NaN loss detected in epoch {epoch_num}. Skipping batch.")
                continue

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            batch_num_edges = (targets != PAD_VALUE_LABELS).sum().item()

            if batch_num_edges > 0:
                total_loss += batch_loss * batch_num_edges
                num_samples += batch_num_edges
                avg_loss = total_loss / num_samples
            else:
                 avg_loss = total_loss / (num_samples + 1e-6)

            loader_pbar.set_postfix(
                avg_loss=f"{avg_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}"
            )

        except Exception as e:
            print(f"\nError during training batch in Epoch {epoch_num}: {e}")
            traceback.print_exc()
            continue

    return total_loss / num_samples if num_samples > 0 else 0.0


def train(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device, label_encoder):
    """Main training loop with tqdm and checkpointing for single-pass model."""

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    start_epoch, best_val_loss, epochs_no_improve = utils.load_checkpoint(
        args.resume_from, model, optimizer, scheduler, device
    )

    print("\n--- Starting/Resuming Training (Single Pass Graph Sequence Mode) ---")
    training_start_time = time.time()

    epoch_pbar = tqdm(range(start_epoch, args.epochs + 1), desc="Epochs", initial=start_epoch-1, total=args.epochs)

    last_completed_epoch = start_epoch - 1

    try:
        for epoch in epoch_pbar:
            epoch_start_time = time.time()

            # Training Phase
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)

            # Validation Phase
            if val_loader:
                val_loss, val_accuracy, _, _ = utils.evaluate(model, val_loader, criterion, device, desc="Validating")
            else:
                val_loss = float('inf')
                val_accuracy = 0.0

            epoch_duration = time.time() - epoch_start_time

            # Logging
            if val_loader:
                log_msg = (f"Epoch {epoch}/{args.epochs} | Time: {epoch_duration:.2f}s | "
                           f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                           f"Val Acc: {val_accuracy:.4f}")
            else:
                log_msg = (f"Epoch {epoch}/{args.epochs} | Time: {epoch_duration:.2f}s | "
                           f"Train Loss: {train_loss:.4f} | (No Validation)")
            epoch_pbar.write(log_msg)

            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss if val_loader else None)
            history['val_accuracy'].append(val_accuracy if val_loader else None)

            # Learning Rate Scheduling
            if val_loader and scheduler:
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < current_lr:
                     epoch_pbar.write(f"📉 Learning rate reduced to {new_lr:.1e}")
            elif scheduler:
                epoch_pbar.write("Info: No validation loader for LR scheduler step. LR may not decrease.")

            # Checkpoint Saving (Best Model based on Val Loss)
            if val_loader and val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                epoch_pbar.write(f"✨ Val Loss Improved by {improvement:.4f} to {val_loss:.4f}. Saving best model... 💾")
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.best_model_path)
                epochs_no_improve = 0
            elif val_loader:
                epochs_no_improve += 1
                epoch_pbar.write(f"📉 No improvement in validation loss for {epochs_no_improve} epoch(s).")


            # Checkpoint Saving (Latest - Every Epoch)
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler and hasattr(scheduler, 'state_dict') else None,
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve,
                'args': vars(args)
            }
            utils.save_checkpoint(checkpoint_state, args.latest_ckpt_path)

            last_completed_epoch = epoch

            # Early Stopping
            if val_loader and args.early_stop > 0 and epochs_no_improve >= args.early_stop:
                epoch_pbar.write(f"\n🛑 Early stopping triggered after {args.early_stop} epochs with no improvement.")
                break

    except KeyboardInterrupt:
        epoch_pbar.write("\nTraining interrupted by user.")
    except Exception as e:
        epoch_pbar.write(f"\nAn error occurred during training loop: {e}")
        traceback.print_exc()
    finally:
        epoch_pbar.close()
        total_training_time = time.time() - training_start_time
        print(f"\nFinished training {last_completed_epoch} epochs.")
        print(f"Total training loop time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")

    return history
