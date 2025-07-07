# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import numpy as np
from sklearn.metrics import classification_report
import argparse
import traceback

import config
import model
import data_utils
import utils
import trainer


# Constant for loss ignore index
PAD_VALUE_LABELS = -100

def main():
    script_start_time = time.time()

    # Configuration
    args = config.setup_arg_parser()
    config.ensure_dirs(args)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"\nFeature Columns: {config.FEATURE_COLS}\n")
    print(f"Using Log Weights: True (Implied by data_utils)")
    print(f"Max Class Weight (for log weights): {args.max_class_weight}")
    print(f"\n    Paths    ")
    print(f"Source Data JSON (Train/Val/Test): {args.train_val_data_json_path}")
    print(f"Scaler: {args.scaler_path}")
    print(f"Encoder: {args.encoder_path}")
    print(f"Best Model Save Path: {args.best_model_path}")
    print(f"Latest Checkpoint Save Path: {args.latest_ckpt_path}")
    print(f"Train CM Plot Path: {args.plot_path_train}")
    print(f"Convergence Plot Path: {args.plot_path_convergence}")
    print(f"Evaluation Plot Path (Clean): {args.eval_plot_path}")
    if args.eval_with_noise:
        print(f"Evaluation Plot Path (Noisy): {config.get_noisy_eval_plot_path(args)}")
    if args.resume_from: print(f"Resuming From: {args.resume_from}")
    print(f"Hyperparameter String: {args.hparam_str}")
    print(f"Label Smoothing: {args.label_smoothing if args.label_smoothing > 0.0 else 'DISABLED'}")
    if args.mlp_head_dims:
        print(f"MLP Head Dimensions: {args.mlp_head_dims}")
    else:
        print("MLP Head Dimensions: [Linear Classifier]")
    print(f"-------------")

    # Print Augmentation Status
    if args.apply_geom_noise:
        print(f"\nOnline GEOMETRIC Augmentation ENABLED")
    else:
        print(f"\nOnline GEOMETRIC Augmentation DISABLED")
    print(f"---------------------------------------------")

    try:
        scaler_obj, label_encoder, num_classes = data_utils.load_scaler_encoder(
            args.scaler_path, args.encoder_path
        )
    except FileNotFoundError as e:
        print(f"FATAL: Could not load scaler/encoder. File not found: {e}. Exiting.")
        exit(1)
    except Exception as e:
        print(f"FATAL: Could not load scaler/encoder. Exiting. Error: {e}")
        traceback.print_exc()
        exit(1)

    # Load Data
    print("\nLoading DataLoaders")
    try:
        train_loader, val_loader, class_weights = data_utils.load_and_split_data(
            args=args, scaler=scaler_obj, label_encoder=label_encoder
        )
        if train_loader is None:
            print("FATAL: Training data loader could not be created. Exiting.")
            exit(1)
        print(f"Class weights loaded: {class_weights.cpu().numpy().round(4)}")
    except FileNotFoundError as e:
        print(f"FATAL: Could not load training/validation data from JSON. File not found: {e}. Exiting.")
        exit(1)
    except Exception as e:
        print(f"FATAL: Could not load training/validation data. Exiting. Error: {e}")
        traceback.print_exc()
        exit(1)

    # Initialize Model
    input_dim = len(config.FEATURE_COLS)
    print(f"\nInitializing TransformerEdgeClassifier (Input Dim: {input_dim})...")
    classifier_model = utils.create_model_from_args(args, num_classes, input_dim).to(DEVICE)
    print(f"Model Trainable Params: {sum(p.numel() for p in classifier_model.parameters() if p.requires_grad):,}")

    # Setup Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE),
        ignore_index=PAD_VALUE_LABELS,
        label_smoothing=args.label_smoothing
    )
    optimizer = optim.Adam(classifier_model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience)

    # Load Checkpoint
    start_epoch, best_val_loss, epochs_no_improve = utils.load_checkpoint(
        args.resume_from, classifier_model, optimizer, scheduler, DEVICE
    )
    classifier_model.to(DEVICE) # Ensure model is on the correct device

    # Run Training
    history = trainer.train(
        args=args, model=classifier_model,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=DEVICE, label_encoder=label_encoder
    )

    # Plot Convergence Curves
    if history:
        print("\nPlotting Training Convergence")
        utils.plot_convergence(
            history=history, output_path=args.plot_path_convergence,
            title=f"Convergence ({args.hparam_str})"
        )
    else:
        print("\nSkipping convergence plotting (no history returned)")

    # --- Final Test Set Evaluation ---
    print("\nFinal Evaluation on Test Set (Using Clean Data)")
    print(f"Loading best model from {args.best_model_path}...")
    model_to_eval = None
    try:
        best_model = utils.create_model_from_args(args, num_classes, input_dim).to(DEVICE)

        state_dict = torch.load(args.best_model_path, map_location=DEVICE)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        missing_keys, unexpected_keys = best_model.load_state_dict(state_dict, strict=True)
        if missing_keys: print(f"Warning: Missing keys loading best model: {missing_keys}")
        if unexpected_keys: print(f"Warning: Unexpected keys loading best model: {unexpected_keys}")
        print("Best model weights loaded successfully.")
        model_to_eval = best_model
        model_to_eval.eval()

    except FileNotFoundError:
        print(f"Warning: Best model '{args.best_model_path}' not found. Using final model from training.")
        model_to_eval = classifier_model # Fallback to the model currently in memory
        if model_to_eval: model_to_eval.eval()
    except Exception as e:
        print(f"Warning: Error loading best model ({e}). Using final model from training.")
        traceback.print_exc()
        model_to_eval = classifier_model # Fallback
        if model_to_eval: model_to_eval.eval()

    # Load Test Data
    print(f"Loading CLEAN test data specified by {args.eval_data_json_path}...")
    try:
        # Use eval scaler/encoder paths defined in config
        scaler_test, encoder_test, _ = data_utils.load_scaler_encoder(args.eval_scaler_path, args.eval_encoder_path)
        target_names = encoder_test.classes_.astype(str) # Get target names here
    except Exception as e:
        print(f"FATAL: Cannot load scaler/encoder for evaluation. Error: {e}")
        traceback.print_exc()
        exit(1)

    test_loader = data_utils.load_test_data(
        args=args, scaler=scaler_test, label_encoder=encoder_test, apply_noise=False
    )

    # Perform CLEAN Evaluation
    if test_loader and model_to_eval:
        criterion_test = nn.CrossEntropyLoss(ignore_index=PAD_VALUE_LABELS)
        test_loss, test_accuracy, test_preds, test_targets = utils.evaluate(
            model=model_to_eval, loader=test_loader, criterion=criterion_test,
            device=DEVICE, desc="Testing (Clean)"
        )

        # Reporting and Plotting
        if not test_targets:
             print("Error: No valid predictions generated for the CLEAN test set.")
        else:
            test_targets_np = np.array(test_targets)
            test_preds_np = np.array(test_preds)
            print(f"\nCLEAN Test Loss: {test_loss:.4f}")
            print(f"CLEAN Test Accuracy: {test_accuracy:.4f}")
            print("\nCLEAN Test Set Classification Report:")
            present_labels = np.unique(np.concatenate((test_targets_np, test_preds_np))).astype(int)
            valid_present_labels = present_labels[present_labels < len(target_names)]

            if len(valid_present_labels) == 0:
                 print("Warning: No valid labels found in clean test predictions/targets. Cannot generate report.")
            else:
                if len(valid_present_labels) < len(present_labels):
                    invalid_labels = present_labels[~np.isin(present_labels, valid_present_labels)]
                    print(f"Warning (Clean): Some predicted/target labels ({invalid_labels}) are outside known classes.")
                target_names_filtered = [name for i, name in enumerate(target_names) if i in valid_present_labels]
                report_labels = valid_present_labels
                if report_labels.size > 0 and len(target_names_filtered) > 0:
                    print(classification_report(
                        test_targets_np, test_preds_np,
                        labels=report_labels, target_names=target_names_filtered,
                        zero_division=0
                    ))
                else: print("Cannot generate classification report - no valid labels or filtered names.")

            utils.plot_confusion_matrix(
                targets=test_targets_np, preds=test_preds_np, class_names=target_names,
                output_path=args.eval_plot_path,
                title=f'Confusion Matrix - CLEAN Test Set\n({args.hparam_str})',
                output_cm_data_path=args.eval_cm_data_path
            )
    else:
         if not test_loader: print("\n--- Clean test set data not loaded. Skipping clean test evaluation. ---")
         if not model_to_eval: print("\n--- Model for evaluation not available. Skipping clean test evaluation. ---")


    # NOISY EVALUATION Section
    if args.eval_with_noise:
        print("\n--- Evaluation on Test Set (Applying GEOMETRIC Noise) ---")
        print(f"NOTE: Using geometric noise parameters defined during training setup (args).")

        noisy_test_loader = data_utils.load_test_data(
            args=args, scaler=scaler_test, label_encoder=encoder_test, apply_noise=True
        )

        if noisy_test_loader and model_to_eval:
            noisy_test_loss, noisy_test_accuracy, noisy_test_preds, noisy_test_targets = utils.evaluate(
                model=model_to_eval, loader=noisy_test_loader, criterion=criterion_test,
                device=DEVICE, desc="Testing (Noisy)"
            )

            # Reporting and Plotting
            if not noisy_test_targets:
                 print("Error: No valid predictions generated for the NOISY test set.")
            else:
                noisy_test_targets_np = np.array(noisy_test_targets)
                noisy_test_preds_np = np.array(noisy_test_preds)
                print(f"\nNOISY Test Loss: {noisy_test_loss:.4f}")
                print(f"NOISY Test Accuracy: {noisy_test_accuracy:.4f}")
                print("\nNOISY Test Set Classification Report:")
                present_labels_noisy = np.unique(np.concatenate((noisy_test_targets_np, noisy_test_preds_np))).astype(int)
                valid_present_labels_noisy = present_labels_noisy[present_labels_noisy < len(target_names)]

                if len(valid_present_labels_noisy) == 0:
                    print("Warning: No valid labels found in NOISY test predictions/targets. Cannot generate report.")
                else:
                    if len(valid_present_labels_noisy) < len(present_labels_noisy):
                        invalid_labels_noisy = present_labels_noisy[~np.isin(present_labels_noisy, valid_present_labels_noisy)]
                        print(f"Warning (Noisy): Some predicted/target labels ({invalid_labels_noisy}) are outside known classes.")
                    target_names_filtered_noisy = [name for i, name in enumerate(target_names) if i in valid_present_labels_noisy]
                    report_labels_noisy = valid_present_labels_noisy
                    if report_labels_noisy.size > 0 and len(target_names_filtered_noisy) > 0:
                        print(classification_report(
                            noisy_test_targets_np, noisy_test_preds_np,
                            labels=report_labels_noisy, target_names=target_names_filtered_noisy,
                            zero_division=0
                        ))
                    else: print("Cannot generate classification report - no valid labels or filtered names for noisy data.")

                noisy_eval_plot_path = config.get_noisy_eval_plot_path(args)

                utils.plot_confusion_matrix(
                    targets=noisy_test_targets_np, preds=noisy_test_preds_np, class_names=target_names,
                    output_path=noisy_eval_plot_path,
                    title=f'Confusion Matrix - NOISY Test Set\n({args.hparam_str})'
                )
        else:
             if not noisy_test_loader: print("\n--- Noisy test set data not loaded. Skipping noisy test evaluation. ---")
             if not model_to_eval: print("\n--- Model for noisy evaluation not available (check best model loading). Skipping noisy test evaluation. ---")


    # Finish
    script_end_time = time.time()
    total_script_duration = script_end_time - script_start_time
    print("\n--- Main Script Finished ---")
    print(f"Total execution time: {total_script_duration:.2f} seconds ({total_script_duration/60:.2f} minutes)")
    print(f"Hyperparameters Used: {args.hparam_str}")
    print(f"Best model saved to (if improved): {args.best_model_path}")
    print(f"Clean evaluation plot saved to: {args.eval_plot_path}")
    if args.eval_with_noise:
        print(f"Noisy evaluation plot saved to: {config.get_noisy_eval_plot_path(args)}")

if __name__ == "__main__":
    main()
