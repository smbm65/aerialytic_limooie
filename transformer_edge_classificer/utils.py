# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os
from tqdm import tqdm
import math

import model


# Must match data_utils
PAD_VALUE_LABELS = -100

# HELPER FUNCTION
def create_model_from_args(args, num_classes, input_dim):
    """
    Centralized function to initialize the model architecture from args.
    This prevents code duplication in main.py, evaluate.py, etc.
    """
    return model.TransformerEdgeClassifier(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        num_classes=num_classes,
        dropout=args.dropout,
        norm_first=args.norm_first,
        k_nearest=args.k_nearest,
        mlp_head_dims=args.mlp_head_dims
    )


def evaluate(model, loader, criterion, device, desc="Evaluating"):
    """Evaluates the sequence model on the provided data loader (single pass)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_samples = 0

    if loader is None:
        print(f"Warning: {desc} loader is None. Skipping evaluation.")
        return 0.0, 0.0, [], []

    loader_pbar = tqdm(loader, desc=desc, leave=False)

    num_classes = model.num_classes

    with torch.no_grad():
        for features, targets, attention_mask in loader_pbar:
            features = features.to(device)
            targets = targets.to(device)
            attention_mask = attention_mask.to(device)

            try:
                final_logits = model(features, attention_mask)

                loss = criterion(final_logits.view(-1, num_classes), targets.view(-1))

                mask_valid = (targets != PAD_VALUE_LABELS)

                logits_flat = final_logits.view(-1, num_classes)
                targets_flat = targets.view(-1)
                mask_valid_flat = mask_valid.view(-1)

                valid_logits = logits_flat[mask_valid_flat]
                valid_targets = targets_flat[mask_valid_flat]

                if valid_targets.numel() > 0:
                     batch_loss = loss.item()
                     batch_num_edges = valid_targets.numel()

                     total_loss += batch_loss * batch_num_edges
                     num_samples += batch_num_edges

                     preds = torch.argmax(valid_logits, dim=1)

                     all_preds.extend(preds.cpu().numpy())
                     all_targets.extend(valid_targets.cpu().numpy())

                     current_avg_loss = total_loss / num_samples if num_samples > 0 else 0
                     loader_pbar.set_postfix(avg_loss=f"{current_avg_loss:.4f}")

            except Exception as e:
                 print(f"\nError during {desc} batch processing: {e}")
                 import traceback
                 traceback.print_exc()
                 continue

    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets)) if num_samples > 0 else 0.0

    if num_samples == 0:
        all_preds = []
        all_targets = []

    return avg_loss, accuracy, all_preds, all_targets


def plot_convergence(history, output_path, title="Training Convergence"):
    """Plots training/validation loss and validation accuracy curves."""
    print(f"\nPlotting convergence curves to {output_path}...")
    if not history or not history.get('train_loss') or not history.get('val_loss') or not history.get('val_accuracy'):
        print("Warning: History dictionary is incomplete. Skipping convergence plot.")
        return
    if not any(history['train_loss']) and not any(l is not None for l in history['val_loss']) and not any(a is not None for a in history['val_accuracy']):
        print("Warning: History lists are empty or contain only None. Skipping convergence plot.")
        return

    epochs = range(1, len(history['train_loss']) + 1)
    valid_epochs_val = [i + 1 for i, v in enumerate(history['val_loss']) if v is not None]
    valid_val_loss = [v for v in history['val_loss'] if v is not None]
    valid_epochs_acc = [i + 1 for i, v in enumerate(history['val_accuracy']) if v is not None]
    valid_val_acc = [v for v in history['val_accuracy'] if v is not None]

    fig = None
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title, fontsize=14)

        ax1.plot(epochs, history['train_loss'], color='blue', linestyle='-', label='Training Loss')
        if valid_val_loss:
            ax1.plot(valid_epochs_val, valid_val_loss, color='red', linestyle='-', label='Validation Loss')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.tick_params(axis='x', direction='in')
        ax1.set_ylim(bottom=0)

        if valid_val_acc:
             ax2.plot(valid_epochs_acc, valid_val_acc, color='green', linestyle='-', label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        if valid_val_acc: ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_ylim(0, 1.05)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plot_dir = os.path.dirname(output_path)
        if plot_dir: os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Convergence plot saved successfully.")

    except Exception as e:
        print(f"Error generating convergence plot: {e}")
    finally:
        if fig is not None and plt.fignum_exists(fig.number):
             plt.close(fig)


def plot_confusion_matrix(targets, preds, class_names, output_path, title='Confusion Matrix', output_cm_data_path=None):
    """Plots and saves the confusion matrix."""
    if not isinstance(targets, (list, np.ndarray)) or not isinstance(preds, (list, np.ndarray)):
         print("Warning: Cannot plot confusion matrix. Inputs must be lists or numpy arrays.")
         return
    if len(targets) == 0 or len(preds) == 0:
         print("Warning: Cannot plot confusion matrix. Empty targets or predictions provided.")
         return
    if len(targets) != len(preds):
        print(f"Warning: Cannot plot confusion matrix. Targets ({len(targets)}) and predictions ({len(preds)}) have different lengths.")
        return

    present_labels = np.unique(np.concatenate((targets, preds)))
    max_label = int(max(present_labels)) if len(present_labels) > 0 else -1
    if max_label >= len(class_names):
         print(f"Warning: Max label index ({max_label}) >= number of class names ({len(class_names)}). CM labels might be incorrect.")
         display_labels = np.arange(max_label + 1)
    else:
         display_labels = class_names

    print(f"\nPlotting confusion matrix and saving to {output_path}...")
    fig = None
    try:
        cm = confusion_matrix(targets, preds, labels=np.arange(len(class_names)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

        if output_cm_data_path:
            try:
                cm_data_dir = os.path.dirname(output_cm_data_path)
                if cm_data_dir: os.makedirs(cm_data_dir, exist_ok=True)
                np.save(output_cm_data_path, cm)
                print(f"Numerical CM data saved to {output_cm_data_path}")
            except Exception as e_save_cm:
                print(f"Warning: Could not save numerical CM data to {output_cm_data_path}: {e_save_cm}")

        fig_width = max(8, len(class_names) * 0.6)
        fig_height = max(6, len(class_names) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', values_format='d')
        ax.set_title(title, fontsize=14)
        plt.tight_layout()

        plot_dir = os.path.dirname(output_path)
        if plot_dir: os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved successfully to {output_path}")

    except Exception as e:
        print(f"Error generating confusion matrix plot: {e}")
    finally:
        if fig is not None and plt.fignum_exists(fig.number):
             plt.close(fig)


def save_checkpoint(state, filepath):
    """Saves model and training parameters at checkpoint."""
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        temp_filepath = filepath + ".tmp"
        torch.save(state, temp_filepath)
        os.replace(temp_filepath, filepath)
    except Exception as e:
        print(f"\n⚠️ Error saving checkpoint to {filepath}: {e}")
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except OSError:
                pass


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    """Loads model parameters (and optionally optimizer/scheduler state) from checkpoint."""
    start_epoch = 1
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if filepath and os.path.exists(filepath):
        print(f"\nLoading checkpoint: {filepath}")
        try:
            checkpoint = torch.load(filepath, map_location=device)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
                    if missing_keys: print(f"Warning: Missing keys found in model state dict: {missing_keys}")
                    if unexpected_keys: print(f"Warning: Unexpected keys found in model state dict: {unexpected_keys}")
                    if not missing_keys and not unexpected_keys: print("Model state dict loaded successfully (strict).")
                except RuntimeError as e:
                     print(f"Warning: Could not load model state dict strictly: {e}. Trying non-strict...")
                     missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                     if missing_keys: print(f"  Warning (non-strict): Missing keys found: {missing_keys}")
                     if unexpected_keys: print(f"  Warning (non-strict): Unexpected keys found: {unexpected_keys}")
                     if not missing_keys and not unexpected_keys: print("  Model state dict loaded successfully (non-strict).")
                except Exception as e:
                     print(f"Error loading model state dict: {e}.")
            else:
                print("Warning: Checkpoint does not contain 'model_state_dict'. Model weights not loaded.")

            if optimizer and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                    print("Optimizer state loaded.")
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {e}")
            elif optimizer:
                print("Warning: Checkpoint does not contain 'optimizer_state_dict', but optimizer was provided.")

            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                 try:
                     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                     print("Scheduler state loaded.")
                 except Exception as e:
                     print(f"Warning: Could not load scheduler state: {e}. Scheduler will start from defaults.")
            elif scheduler:
                print("Warning: Checkpoint missing 'scheduler_state_dict' or it was None, but scheduler was provided.")

            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            saved_args = checkpoint.get('args', None)
            if saved_args:
                 print(f"Checkpoint saved with args (example): lr={saved_args.get('lr', 'N/A')}, bs={saved_args.get('batch_size', 'N/A')}")
            print(f"Resuming training from Epoch {start_epoch}.")
            print(f"Previous Best Val Loss: {best_val_loss:.4f}. Epochs w/o Improvement: {epochs_no_improve}")

        except Exception as e:
            print(f"ERROR loading checkpoint '{filepath}': {e}. Training will start from scratch.")
            start_epoch = 1
            best_val_loss = float('inf')
            epochs_no_improve = 0
    else:
        if filepath:
             print(f"\nCheckpoint file not found: '{filepath}'. Starting training from scratch.")
        else:
             print("\nNo checkpoint path provided. Starting training from scratch.")

    return start_epoch, best_val_loss, epochs_no_improve
