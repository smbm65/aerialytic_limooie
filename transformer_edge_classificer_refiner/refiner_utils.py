# refiner_utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import os
from tqdm import tqdm

import refiner_data_utils as r_du


PAD_VALUE_LABELS_REF = r_du.PAD_VALUE_LABELS_REF
PAD_VALUE_ORIG_PRED_IDX_REF = r_du.PAD_VALUE_ORIG_PRED_IDX_REF
PAD_VALUE_CORRECTNESS_REF = r_du.PAD_VALUE_CORRECTNESS_REF


def evaluate_refiner(model, loader, main_criterion, correctness_criterion, device, args, desc="Evaluating Refiner"):
    model.eval()
    total_main_loss = 0.0
    total_correctness_loss = 0.0
    total_combined_loss = 0.0

    main_preds_eval = []
    main_targets_eval = []
    correctness_preds_eval = []
    correctness_targets_eval = []
    overall_refined_preds = []
    overall_gt_targets = []

    num_valid_main_samples_for_loss = 0
    num_valid_correctness_samples_for_loss = 0
    num_valid_overall_samples = 0

    if loader is None:
        print(f"Warning: Refiner loader for {desc} is None. Skipping.")
        # Return structure matching expected output (12 elements)
        return 0.0, 0.0, 0.0, 0.0, 0.0, [], [], [], [], [], [], 0.0

    loader_pbar = tqdm(loader, desc=desc, leave=False, ncols=100)

    with torch.no_grad():
        for features, gt_labels, orig_pred_indices, correctness_targets, attention_mask in loader_pbar:
            features = features.to(device)
            gt_labels = gt_labels.to(device)
            orig_pred_indices = orig_pred_indices.to(device)
            correctness_targets = correctness_targets.to(device)
            attention_mask = attention_mask.to(device)

            main_logits, correctness_logits = model(features, attention_mask) # attention_mask is key_padding_mask

            # Mask for valid (non-padded) elements
            valid_elements_mask = ~attention_mask # Padded elements are True in attention_mask

            # 1. Main Classification Head Evaluation
            # Loss only on originally incorrect & valid items
            main_loss_mask = (orig_pred_indices != gt_labels) & valid_elements_mask & (orig_pred_indices != PAD_VALUE_ORIG_PRED_IDX_REF)
            
            loss_main = torch.tensor(0.0, device=device)
            if main_loss_mask.any():
                main_logits_for_loss = main_logits[main_loss_mask]
                gt_labels_for_loss = gt_labels[main_loss_mask]
                if main_logits_for_loss.numel() > 0 and gt_labels_for_loss.numel() > 0:
                    loss_main = main_criterion(main_logits_for_loss, gt_labels_for_loss)
                    total_main_loss += loss_main.item() * main_loss_mask.sum().item()
                    num_valid_main_samples_for_loss += main_loss_mask.sum().item()

                    preds_main_subset = torch.argmax(main_logits_for_loss, dim=1)
                    main_preds_eval.extend(preds_main_subset.cpu().numpy())
                    main_targets_eval.extend(gt_labels_for_loss.cpu().numpy())
            
            # For overall refined accuracy (all valid elements)
            if valid_elements_mask.any():
                main_logits_overall = main_logits[valid_elements_mask]
                gt_labels_overall = gt_labels[valid_elements_mask]
                if main_logits_overall.numel() > 0:
                    preds_overall = torch.argmax(main_logits_overall, dim=1)
                    overall_refined_preds.extend(preds_overall.cpu().numpy())
                    overall_gt_targets.extend(gt_labels_overall.cpu().numpy())
                    num_valid_overall_samples += valid_elements_mask.sum().item()

            # 2. Correctness Prediction Head Evaluation
            # Correctness targets are already float, padded with PAD_VALUE_CORRECTNESS_REF
            correctness_loss_mask = (correctness_targets != PAD_VALUE_CORRECTNESS_REF) & valid_elements_mask
            loss_correctness = torch.tensor(0.0, device=device)
            if correctness_loss_mask.any():
                correctness_logits_squeezed = correctness_logits.squeeze(-1) # [B, S]
                correctness_logits_for_loss = correctness_logits_squeezed[correctness_loss_mask]
                correctness_targets_for_loss = correctness_targets[correctness_loss_mask]

                if correctness_logits_for_loss.numel() > 0 and correctness_targets_for_loss.numel() > 0:
                    loss_correctness = correctness_criterion(correctness_logits_for_loss, correctness_targets_for_loss)
                    total_correctness_loss += loss_correctness.item() * correctness_loss_mask.sum().item()
                    num_valid_correctness_samples_for_loss += correctness_loss_mask.sum().item()

                    preds_corr = (torch.sigmoid(correctness_logits_for_loss) > 0.5).long()
                    correctness_preds_eval.extend(preds_corr.cpu().numpy())
                    correctness_targets_eval.extend(correctness_targets_for_loss.cpu().numpy().astype(int)) # Convert to int for metrics

            # 3. Combined Loss (for reference)
            main_lw = args.main_loss_weight if hasattr(args, 'main_loss_weight') else 1.0
            corr_lw = args.correctness_loss_weight if hasattr(args, 'correctness_loss_weight') else 0.5
            
            # Sum of weighted raw losses (not yet averaged over epoch)
            # These are per-batch average losses multiplied by number of contributing samples
            current_batch_main_contrib = main_loss_mask.sum().item()
            current_batch_corr_contrib = correctness_loss_mask.sum().item()

            total_combined_loss += (main_lw * loss_main.item() * current_batch_main_contrib + \
                                   corr_lw * loss_correctness.item() * current_batch_corr_contrib)

    avg_main_loss = total_main_loss / num_valid_main_samples_for_loss if num_valid_main_samples_for_loss > 0 else 0.0
    main_accuracy_on_subset = accuracy_score(main_targets_eval, main_preds_eval) if main_targets_eval else 0.0
    
    avg_correctness_loss = total_correctness_loss / num_valid_correctness_samples_for_loss if num_valid_correctness_samples_for_loss > 0 else 0.0
    correctness_accuracy = accuracy_score(correctness_targets_eval, correctness_preds_eval) if correctness_targets_eval else 0.0

    total_involved_samples_for_combined = max(num_valid_main_samples_for_loss, num_valid_correctness_samples_for_loss, num_valid_overall_samples, 1)
    avg_combined_loss = total_combined_loss / total_involved_samples_for_combined

    overall_refined_accuracy = accuracy_score(overall_gt_targets, overall_refined_preds) if overall_gt_targets else 0.0
    
    return (avg_main_loss, main_accuracy_on_subset,
            avg_correctness_loss, correctness_accuracy,
            avg_combined_loss,
            main_preds_eval, main_targets_eval,
            correctness_preds_eval, correctness_targets_eval,
            overall_refined_preds, overall_gt_targets,
            overall_refined_accuracy)


def plot_convergence_refiner(history, output_path, title="Refiner Training Convergence"):
    print(f"\nPlotting refiner convergence curves to {output_path}...")
    if not history: 
        print("Warning: History is empty. Skipping plot.")
        return
    epochs = range(1, len(history.get('train_main_loss', [])) + 1)
    if not epochs: 
        print("Warning: No epoch data in history. Skipping plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(title, fontsize=16)

    ax1 = axes[0]

    if history.get('train_main_loss'): ax1.plot(epochs, history['train_main_loss'], 'b-', label='Train Main Loss')
    if history.get('val_main_loss') and any(v is not None for v in history['val_main_loss']): ax1.plot(epochs, history['val_main_loss'], 'b--', label='Val Main Loss')
    
    ax1.set_ylabel('Main Clf Loss')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2 = axes[1]
    
    if history.get('train_corr_loss'): ax2.plot(epochs, history['train_corr_loss'], 'r-', label='Train Correctness Loss')
    if history.get('val_corr_loss') and any(v is not None for v in history['val_corr_loss']): ax2.plot(epochs, history['val_corr_loss'], 'r--', label='Val Correctness Loss')
    
    ax2.set_ylabel('Correctness Loss')
    ax2.legend(loc='best')
    ax2.grid(True, linestyle=':', alpha=0.7)

    ax3 = axes[2]
    
    if history.get('train_main_acc'): ax3.plot(epochs, history['train_main_acc'], 'g-', label='Train Main Acc (subset)')
    if history.get('val_main_acc') and any(v is not None for v in history['val_main_acc']): ax3.plot(epochs, history['val_main_acc'], 'g--', label='Val Main Acc (subset)')
    if history.get('train_corr_acc'): ax3.plot(epochs, history['train_corr_acc'], 'm-', label='Train Correctness Acc')
    if history.get('val_corr_acc') and any(v is not None for v in history['val_corr_acc']): ax3.plot(epochs, history['val_corr_acc'], 'm--', label='Val Correctness Acc')
    if history.get('val_overall_refined_acc') and any(v is not None for v in history['val_overall_refined_acc']): ax3.plot(epochs, history['val_overall_refined_acc'], 'c-', label='Val Overall Refined Acc', linewidth=2)
    
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend(loc='best')
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_dir = os.path.dirname(output_path)
    if plot_dir: os.makedirs(plot_dir, exist_ok=True)
    try:
        plt.savefig(output_path, dpi=150)
        print(f"Refiner convergence plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving convergence plot: {e}")
    finally:
        plt.close(fig)


def plot_confusion_matrix_refiner(targets, preds, class_names, output_path, title='CM', output_cm_data_path=None, is_binary_correctness=False):
    if not isinstance(targets, (list, np.ndarray)) or not isinstance(preds, (list, np.ndarray)) or \
       len(targets) == 0 or len(preds) == 0 or len(targets) != len(preds):
        print(f"Warning: Cannot plot CM for '{title}'. Invalid inputs (targets: {len(targets)}, preds: {len(preds)}).")
        return
    
    print(f"\nPlotting CM for '{title}' and saving to {output_path}...")
    fig = None
    try:
        labels_for_cm = [0, 1] if is_binary_correctness else np.arange(len(class_names))
        display_labels_for_cm = ['Orig Wrong', 'Orig Correct'] if is_binary_correctness else class_names
        
        cm = confusion_matrix(targets, preds, labels=labels_for_cm)

        if output_cm_data_path:
            try:
                cm_data_dir = os.path.dirname(output_cm_data_path)
                if cm_data_dir: os.makedirs(cm_data_dir, exist_ok=True)
                np.save(output_cm_data_path, cm)
                print(f"  Numerical CM data saved to {output_cm_data_path}")
            except Exception as e_save: print(f"  Warning: Could not save numerical CM: {e_save}")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels_for_cm)
        fig_width = max(8, len(display_labels_for_cm) * 0.8 + 2)
        fig_height = max(6, len(display_labels_for_cm) * 0.6 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', values_format='d')
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        plot_dir = os.path.dirname(output_path)
        if plot_dir: os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"  Plot saved successfully to {output_path}")
    except Exception as e: print(f"  Error generating CM plot for '{title}': {e}")
    finally:
        if fig and plt.fignum_exists(fig.number): plt.close(fig)


def save_checkpoint_refiner(state, filepath):
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        temp_filepath = filepath + ".tmp_refiner"
        torch.save(state, temp_filepath)
        os.replace(temp_filepath, filepath)
    except Exception as e:
        print(f"\n⚠️ Error saving refiner checkpoint to {filepath}: {e}")
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            try: os.remove(temp_filepath)
            except OSError: pass


def load_checkpoint_refiner(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    start_epoch = 1
    best_val_combined_loss = float('inf')
    epochs_no_improve = 0

    if filepath and os.path.exists(filepath):
        print(f"\nLoading refiner checkpoint: {filepath}")
        try:
            checkpoint = torch.load(filepath, map_location=device)
            if 'model_state_dict' in checkpoint:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if missing_keys: print(f"  Warn (Refiner Load): Missing keys: {missing_keys}")
                if unexpected_keys: print(f"  Warn (Refiner Load): Unexpected keys: {unexpected_keys}")
                print("  Refiner model state loaded.")
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("  Refiner optimizer state loaded.")
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("  Refiner scheduler state loaded.")
            
            start_epoch = checkpoint.get('epoch', 0) + 1 # Resume from next epoch
            best_val_combined_loss = checkpoint.get('best_val_combined_loss', float('inf'))
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            print(f"  Resuming refiner training from Epoch {start_epoch}. Best Val Comb Loss: {best_val_combined_loss:.4f}")
        except Exception as e:
            print(f"  ERROR loading refiner checkpoint '{filepath}': {e}. Starting refiner from scratch.")
            start_epoch = 1
            best_val_combined_loss = float('inf')
            epochs_no_improve = 0
    else:
        if filepath: print(f"Refiner checkpoint '{filepath}' not found. Starting from scratch.")
        else: print("No refiner checkpoint specified. Starting from scratch.")
    return start_epoch, best_val_combined_loss, epochs_no_improve
