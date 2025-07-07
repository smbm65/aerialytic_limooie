# refiner_trainer.py
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import refiner_utils as r_utils
import os
import traceback

# Import PAD values from refiner_data_utils
import refiner_data_utils as r_du

PAD_VALUE_LABELS_REF = r_du.PAD_VALUE_LABELS_REF
PAD_VALUE_ORIG_PRED_IDX_REF = r_du.PAD_VALUE_ORIG_PRED_IDX_REF
PAD_VALUE_CORRECTNESS_REF = r_du.PAD_VALUE_CORRECTNESS_REF


def train_one_epoch_refiner(model, loader, main_criterion, correctness_criterion, optimizer, device, epoch_num, total_epochs, args):
    model.train()
    total_main_loss_epoch = 0.0
    total_correctness_loss_epoch = 0.0
    total_combined_loss_epoch_norm = 0.0

    main_preds_list = []
    main_targets_list = []
    correctness_preds_list = []
    correctness_targets_list = []

    num_valid_main_samples_epoch = 0
    num_valid_correctness_samples_epoch = 0
    num_batches_processed = 0

    if loader is None:
        print(f"Warning: Refiner training loader for Epoch {epoch_num} is None. Skipping.")
        return {'train_main_loss': 0.0, 'train_main_acc': 0.0, 'train_corr_loss': 0.0, 'train_corr_acc': 0.0, 'train_comb_loss': 0.0}

    loader_pbar = tqdm(
        loader, desc=f"Train Epoch {epoch_num}/{total_epochs}",
        leave=False,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )

    for features, gt_labels, orig_pred_indices, correctness_targets_float, attention_mask in loader_pbar:
        features, gt_labels, orig_pred_indices, correctness_targets_float, attention_mask = \
            features.to(device), gt_labels.to(device), orig_pred_indices.to(device), correctness_targets_float.to(device), attention_mask.to(device)

        optimizer.zero_grad()
        
        try:
            main_logits, correctness_logits = model(features, attention_mask)
            valid_elements_mask = ~attention_mask

            # Main Classification Loss
            main_loss_mask = (orig_pred_indices != gt_labels) & valid_elements_mask & (orig_pred_indices != PAD_VALUE_ORIG_PRED_IDX_REF)
            loss_main_batch = torch.tensor(0.0, device=device)
            num_main_samples_batch = 0
            if main_loss_mask.any():
                main_logits_for_loss = main_logits[main_loss_mask]
                gt_labels_for_loss = gt_labels[main_loss_mask]
                if main_logits_for_loss.numel() > 0:
                    loss_main_batch = main_criterion(main_logits_for_loss, gt_labels_for_loss)
                    preds_main = torch.argmax(main_logits_for_loss, dim=1)
                    main_preds_list.extend(preds_main.cpu().numpy())
                    main_targets_list.extend(gt_labels_for_loss.cpu().numpy())
                    num_main_samples_batch = gt_labels_for_loss.numel()

            # Correctness Prediction Loss
            correctness_loss_mask = (correctness_targets_float != PAD_VALUE_CORRECTNESS_REF) & valid_elements_mask
            loss_correctness_batch = torch.tensor(0.0, device=device)
            num_corr_samples_batch = 0
            if correctness_loss_mask.any():
                correctness_logits_squeezed = correctness_logits.squeeze(-1)
                correctness_logits_for_loss = correctness_logits_squeezed[correctness_loss_mask]
                correctness_targets_for_loss = correctness_targets_float[correctness_loss_mask]
                if correctness_logits_for_loss.numel() > 0:
                    loss_correctness_batch = correctness_criterion(correctness_logits_for_loss, correctness_targets_for_loss)
                    preds_corr = (torch.sigmoid(correctness_logits_for_loss) > 0.5).long()
                    correctness_preds_list.extend(preds_corr.cpu().numpy())
                    correctness_targets_list.extend(correctness_targets_for_loss.cpu().numpy().astype(int))
                    num_corr_samples_batch = correctness_targets_for_loss.numel()

            # Combined Loss for Backward Pass
            raw_sum_loss_main = loss_main_batch * num_main_samples_batch
            raw_sum_loss_corr = loss_correctness_batch * num_corr_samples_batch
            combined_loss_unnormalized = (args.main_loss_weight * raw_sum_loss_main + args.correctness_loss_weight * raw_sum_loss_corr)
            
            num_total_valid_edges_batch = valid_elements_mask.sum().item()
            combined_loss_for_bp = combined_loss_unnormalized / num_total_valid_edges_batch if num_total_valid_edges_batch > 0 else torch.tensor(0.0, device=device)

            if torch.isnan(combined_loss_for_bp) or torch.isinf(combined_loss_for_bp):
                print(f"\nWarning: NaN/Inf combined loss in epoch {epoch_num}. Skipping batch backward().")
                continue
            
            combined_loss_for_bp.backward()
            optimizer.step()

            # Accumulate stats for epoch-level reporting
            total_main_loss_epoch += raw_sum_loss_main.item()
            total_correctness_loss_epoch += raw_sum_loss_corr.item()
            total_combined_loss_epoch_norm += combined_loss_for_bp.item()
            num_valid_main_samples_epoch += num_main_samples_batch
            num_valid_correctness_samples_epoch += num_corr_samples_batch
            num_batches_processed += 1

            # Update progress bar postfix with running averages
            avg_main_loss_so_far = total_main_loss_epoch / num_valid_main_samples_epoch if num_valid_main_samples_epoch > 0 else 0
            avg_corr_loss_so_far = total_correctness_loss_epoch / num_valid_correctness_samples_epoch if num_valid_correctness_samples_epoch > 0 else 0
            main_acc_so_far = np.mean(np.array(main_preds_list) == np.array(main_targets_list)) if main_targets_list else 0
            corr_acc_so_far = np.mean(np.array(correctness_preds_list) == np.array(correctness_targets_list)) if correctness_targets_list else 0
            
            loader_pbar.set_postfix(
                mL=f"{avg_main_loss_so_far:.3f}", mA=f"{main_acc_so_far:.2f}",
                cL=f"{avg_corr_loss_so_far:.3f}", cA=f"{corr_acc_so_far:.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}"
            )
        
        except Exception as e:
            print(f"\nError during training batch in Epoch {epoch_num}: {e}")
            traceback.print_exc()
            continue

    # Final epoch metrics
    avg_epoch_main_loss = total_main_loss_epoch / num_valid_main_samples_epoch if num_valid_main_samples_epoch > 0 else 0.0
    avg_epoch_correctness_loss = total_correctness_loss_epoch / num_valid_correctness_samples_epoch if num_valid_correctness_samples_epoch > 0 else 0.0
    avg_epoch_combined_loss = total_combined_loss_epoch_norm / num_batches_processed if num_batches_processed > 0 else 0.0
    main_accuracy = np.mean(np.array(main_preds_list) == np.array(main_targets_list)) if main_targets_list else 0.0
    correctness_accuracy = np.mean(np.array(correctness_preds_list) == np.array(correctness_targets_list)) if correctness_targets_list else 0.0
    
    return {
        'train_main_loss': avg_epoch_main_loss, 'train_main_acc': main_accuracy,
        'train_corr_loss': avg_epoch_correctness_loss, 'train_corr_acc': correctness_accuracy,
        'train_comb_loss': avg_epoch_combined_loss
    }


def train_refiner(args, model, train_loader, val_loader, main_criterion, correctness_criterion, optimizer, scheduler, device, label_encoder):
    history = {
        'train_main_loss': [], 'train_main_acc': [], 'train_corr_loss': [], 'train_corr_acc': [], 'train_comb_loss': [],
        'val_main_loss': [], 'val_main_acc': [], 'val_corr_loss': [], 'val_corr_acc': [], 'val_comb_loss': [],
        'val_overall_refined_acc': []
    }

    start_epoch, best_val_combined_loss, epochs_no_improve = r_utils.load_checkpoint_refiner(
        args.resume_from, model, optimizer, scheduler, device
    )
    model.to(device)

    print("\nStarting/Resuming Refiner Training")
    training_start_time = time.time()
    epoch_pbar = tqdm(range(start_epoch, args.epochs + 1), desc="Epochs", initial=start_epoch - 1, total=args.epochs)
    last_completed_epoch = start_epoch - 1

    try:
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            
            # Training Phase
            train_metrics = train_one_epoch_refiner(model, train_loader, main_criterion, correctness_criterion, optimizer, device, epoch, args.epochs, args)
            for key, val in train_metrics.items():
                history[key].append(val)

            # Validation Phase
            val_metrics = {}
            if val_loader:
                (val_main_loss, val_main_acc, val_corr_loss, val_corr_acc, val_comb_loss,
                 _, _, _, _, _, _, val_overall_acc
                ) = r_utils.evaluate_refiner(model, val_loader, main_criterion, correctness_criterion, device, args, desc="Validating")
                val_metrics = {
                    'val_main_loss': val_main_loss, 'val_main_acc': val_main_acc,
                    'val_corr_loss': val_corr_loss, 'val_corr_acc': val_corr_acc,
                    'val_comb_loss': val_comb_loss, 'val_overall_refined_acc': val_overall_acc
                }
            else: # Handle case with no validation loader
                val_metrics = {
                    'val_main_loss': float('inf'), 'val_main_acc': 0.0,
                    'val_corr_loss': float('inf'), 'val_corr_acc': 0.0,
                    'val_comb_loss': float('inf'), 'val_overall_refined_acc': 0.0
                }

            for key, val in val_metrics.items():
                history[key].append(val if val_loader else None)
            
            epoch_duration = time.time() - epoch_start_time

            # Logging
            log_msg = (
                f"Epoch {epoch}/{args.epochs} | Time: {epoch_duration:.1f}s | "
                f"Train Loss (M/C/Comb): {train_metrics['train_main_loss']:.3f}/{train_metrics['train_corr_loss']:.3f}/{train_metrics['train_comb_loss']:.3f} | "
                f"Train Acc (M/C): {train_metrics['train_main_acc']:.3f}/{train_metrics['train_corr_acc']:.3f}"
            )
            if val_loader:
                log_msg += (
                    f"\n           Val Loss (M/C/Comb):   {val_metrics['val_main_loss']:.3f}/{val_metrics['val_corr_loss']:.3f}/{val_metrics['val_comb_loss']:.3f} | "
                    f"Val Acc (M/C/Refined): {val_metrics['val_main_acc']:.3f}/{val_metrics['val_corr_acc']:.3f}/{val_metrics['val_overall_refined_acc']:.3f}"
                )
            else:
                log_msg += "\n           (No Validation)"
            epoch_pbar.write(log_msg)

            # Learning Rate Scheduling
            current_lr = optimizer.param_groups[0]['lr']
            if val_loader and scheduler:
                val_loss_for_scheduler = val_metrics['val_comb_loss']
                scheduler.step(val_loss_for_scheduler)
                if optimizer.param_groups[0]['lr'] < current_lr:
                    epoch_pbar.write(f"📉 Learning rate reduced to {optimizer.param_groups[0]['lr']:.1e}")

            # Checkpoint Saving (Best Model)
            if val_loader and val_metrics['val_comb_loss'] < best_val_combined_loss:
                improvement = best_val_combined_loss - val_metrics['val_comb_loss']
                epoch_pbar.write(f"✨ ValComb_L Improved by {improvement:.4f} to {val_metrics['val_comb_loss']:.4f}. Saving best refiner model... 💾")
                best_val_combined_loss = val_metrics['val_comb_loss']
                torch.save(model.state_dict(), args.best_model_path)
                epochs_no_improve = 0
            elif val_loader:
                epochs_no_improve += 1
                epoch_pbar.write(f"📉 No improvement in validation combined loss for {epochs_no_improve} epoch(s).")
            
            # Checkpoint Saving (Latest)
            checkpoint_state = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler and hasattr(scheduler, 'state_dict') else None,
                'best_val_combined_loss': best_val_combined_loss, 'epochs_no_improve': epochs_no_improve,
                'args': vars(args)
            }
            r_utils.save_checkpoint_refiner(checkpoint_state, args.latest_ckpt_path)
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
        print(f"Total refiner training loop time: {total_training_time:.2f}s ({total_training_time/60:.2f}m)")
    
    return history
