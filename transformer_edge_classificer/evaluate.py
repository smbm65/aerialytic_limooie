# evaluate.py
import torch
import torch.nn as nn
import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import traceback

# Import project modules
import config
import model
import data_utils
import utils

# Constant for loss ignore index
PAD_VALUE_LABELS = -100

def main():
    script_start_time = time.time()

    # Configuration
    args = config.setup_arg_parser()
    config.ensure_dirs(args)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Script")
    print(f"Using device: {DEVICE}")

    eval_mode = "NOISY (Geometric)" if args.eval_with_noise else "CLEAN"
    print(f"Evaluation Mode: {eval_mode}")
    if args.eval_with_noise:
        print("NOTE: Applying geometric noise parameters defined by command-line args.")

    print(f"Evaluating model: {args.eval_model_path}")
    print(f"Using test data JSON: {args.eval_data_json_path}")
    print(f"Using scaler: {args.eval_scaler_path}")
    print(f"Using encoder: {args.eval_encoder_path}")
    plot_path = config.get_noisy_eval_plot_path(args) if args.eval_with_noise else args.eval_plot_path
    print(f"Plotting CM to: {plot_path}")
    print(f"\nFeature Columns: {config.FEATURE_COLS}\n")
    print(f"-------------------------------------------------------------")

    # Load Scaler/Encoder for Evaluation
    try:
        scaler, label_encoder, num_classes = data_utils.load_scaler_encoder(
            args.eval_scaler_path, args.eval_encoder_path
        )
        if scaler is None or label_encoder is None: raise ValueError("Scaler or Encoder failed to load.")
        target_names = label_encoder.classes_.astype(str)
    except Exception as e:
        print(f"FATAL: Could not load scaler/encoder for evaluation. Exiting. Error: {e}")
        traceback.print_exc()
        exit(1)

    # Initialize Model Architecture
    input_dim = len(config.FEATURE_COLS)
    print(f"Initializing model architecture (Input Dim: {input_dim})...")
    try:
        # Use the centralized model creation function
        eval_model = utils.create_model_from_args(args, num_classes, input_dim).to(DEVICE)
    except AttributeError as e:
        print(f"FATAL: Class 'TransformerEdgeClassifier' (or similar) not found or init error: {e}")
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"FATAL: Error initializing model architecture: {e}")
        print(f"Ensure --d_model, --nhead, --k_nearest etc. match the SAVED model being loaded")
        traceback.print_exc()
        exit(1)

    # Load Model Weights
    print(f"Loading model weights from {args.eval_model_path}...")
    try:
        state_dict = torch.load(args.eval_model_path, map_location=DEVICE)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        missing_keys, unexpected_keys = eval_model.load_state_dict(state_dict, strict=True)
        if missing_keys: print(f"Warning: Missing keys loading model: {missing_keys}")
        if unexpected_keys: print(f"Warning: Unexpected keys loading model: {unexpected_keys}")
        print("Model weights loaded successfully.")
        eval_model.eval()
    except FileNotFoundError:
        print(f"FATAL: Model file not found at {args.eval_model_path}. Exiting.")
        exit(1)
    except Exception as e:
        print(f"FATAL: Error loading model weights: {e}. Check architecture compatibility.")
        traceback.print_exc()
        exit(1)

    # Load Test Data
    print(f"Loading {eval_mode} test data from {args.eval_data_json_path}...")
    test_loader = data_utils.load_test_data(
        args=args,
        scaler=scaler,
        label_encoder=label_encoder,
        apply_noise=args.eval_with_noise
    )

    if test_loader is None:
        print("FATAL: Test data could not be loaded. Exiting.")
        exit(1)

    # Perform Evaluation
    print(f"\nEvaluating on {eval_mode} Test Set")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_VALUE_LABELS)
    eval_start_time = time.time()

    test_loss, test_accuracy, test_preds, test_targets = utils.evaluate(
        model=eval_model,
        loader=test_loader,
        criterion=criterion,
        device=DEVICE,
        desc=f"Evaluating {eval_mode} Test Set"
    )
    eval_time = time.time() - eval_start_time
    print(f"Evaluation finished in {eval_time:.2f} seconds.")


    # Display Results
    if not test_targets:
        print("Error: No valid predictions generated for the test set. Cannot evaluate.")
    else:
        test_targets_np = np.array(test_targets)
        test_preds_np = np.array(test_preds)

        print(f"\n--- {eval_mode} Test Set Performance Metrics ---")
        print(f"Test Loss (ignore_index={PAD_VALUE_LABELS}): {test_loss:.4f}")
        print(f"Test Accuracy (on non-padded edges): {test_accuracy:.4f}")

        print(f"\n{eval_mode} Test Set Classification Report:")
        present_labels = np.unique(np.concatenate((test_targets_np, test_preds_np))).astype(int)
        valid_present_labels = present_labels[present_labels < len(target_names)]

        if len(valid_present_labels) == 0:
             print("Warning: No valid labels found in test predictions/targets for known classes. Cannot generate report.")
        else:
            if len(valid_present_labels) < len(present_labels):
                invalid_labels = present_labels[~np.isin(present_labels, valid_present_labels)]
                print(f"Warning ({eval_mode}): Some predicted/target labels ({invalid_labels}) are outside known classes range [0, {len(target_names)-1}).")

            target_names_filtered = [name for i, name in enumerate(target_names) if i in valid_present_labels]
            report_labels_for_sklearn = valid_present_labels

            if report_labels_for_sklearn.size > 0 and len(target_names_filtered) > 0:
                 if len(report_labels_for_sklearn) != len(target_names_filtered):
                      print(f"Warning: Filtered label count ({len(report_labels_for_sklearn)}) differs from filtered name count ({len(target_names_filtered)}). Report might be incomplete.")

                 print(classification_report(
                    test_targets_np,
                    test_preds_np,
                    labels=report_labels_for_sklearn,
                    target_names=target_names_filtered,
                    zero_division=0
                ))
            else: print("Cannot generate classification report - no valid labels or filtered names found.")

        utils.plot_confusion_matrix(
            targets=test_targets_np,
            preds=test_preds_np,
            class_names=target_names,
            output_path=plot_path,
            title=f'Confusion Matrix - {eval_mode} Test Set Eval\n({os.path.basename(args.eval_data_json_path)})',
            output_cm_data_path=args.eval_cm_data_path
        )

    # Script End
    script_end_time = time.time()
    total_script_duration = script_end_time - script_start_time
    print("\nEvaluation Script Finished")
    print(f"Total script execution time: {total_script_duration:.2f} seconds ({total_script_duration/60:.2f} minutes)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nSCRIPT FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        exit(1)
