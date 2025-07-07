# refiner_evaluate.py
import torch
import torch.nn as nn
import time
import os
import numpy as np
from sklearn.metrics import classification_report
import argparse
import traceback

import refiner_config as r_config
import refiner_model as r_model
import refiner_data_utils as r_data_utils
import refiner_utils as r_utils

PAD_VALUE_LABELS_REF = r_data_utils.PAD_VALUE_LABELS_REF

def main_evaluate_refiner():
    script_start_time = time.time()
    args = r_config.setup_arg_parser()
    r_config.args_parsed = args
    # r_config.ensure_dirs(args) # Not strictly needed if only evaluating and saving to existing run dir

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Refiner Model Evaluation Script")
    print(f"Using device: {DEVICE}")
    
    model_load_path = args.eval_model_path # This is correctly constructed by setup_arg_parser
    print(f"Evaluating Refiner Model: {model_load_path}")
    print(f"Refiner Test Data JSON: {args.eval_data_json_path}")
    print(f"Refiner Scaler: {args.eval_scaler_path}") # eval_scaler_path points to args.scaler_path
    print(f"Refiner Encoder: {args.eval_encoder_path}")

    eval_plot_path = args.eval_plot_path # Base path
    eval_cm_data_path = args.eval_cm_data_path

    if args.eval_with_geom_noise:
        print("Geometric Noise for Evaluation: ENABLED")
        eval_plot_path = r_config.get_noisy_eval_plot_path(args, for_refiner=True)
        if eval_cm_data_path:
             eval_cm_data_path = eval_cm_data_path.replace(".npy", "_geom_noisy.npy")
    else:
        print("Geometric Noise for Evaluation: DISABLED")

    print(f"Plotting Overall CM to: {eval_plot_path}")
    
    if eval_cm_data_path: print(f"Saving Overall CM data to: {eval_cm_data_path}")

    print(f"Refiner HParam String (from loaded args): {args.hparam_str}")
    print(f"Refiner Input Feature Mode (from loaded args): {args.refiner_input_feature_mode}")

    try:
        scaler, label_encoder, num_classes = r_data_utils.load_refiner_scaler_encoder(args)
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
    except Exception as e:
        print(f"FATAL: Could not load refiner scaler/encoder. Error: {e}")
        traceback.print_exc()
        exit(1)

    current_input_dim = len(r_config.GEOM_FEATURE_COLS)
    
    if args.refiner_input_feature_mode == "probs_and_onehot":
        current_input_dim += len(r_config.PROB_FEATURE_COLS) + len(r_config.ONEHOT_FEATURE_COLS)
    elif args.refiner_input_feature_mode == "probs_only":
        current_input_dim += len(r_config.PROB_FEATURE_COLS)
    elif args.refiner_input_feature_mode == "onehot_only":
        current_input_dim += len(r_config.ONEHOT_FEATURE_COLS)

    print(f"Refiner Model Input Dimension (selected mode): {current_input_dim} (Scaled Geom + Raw Prob/Onehot)")

    print(f"\nInitializing Refiner Model Architecture (Input Dim: {current_input_dim})...")
    model_to_eval = r_model.RefinerTransformerEdgeClassifier(
        input_dim=current_input_dim, d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_layers, dim_feedforward=args.dim_ff,
        num_classes=num_classes, dropout=args.dropout, norm_first=args.norm_first,
        k_nearest=args.k_nearest, knn_distance_threshold=args.knn_distance_threshold
    ).to(DEVICE)

    print(f"Loading refiner model weights from {model_load_path}...")
    try:
        state_dict = torch.load(model_load_path, map_location=DEVICE)
        
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        missing, unexpected = model_to_eval.load_state_dict(state_dict, strict=False)
        
        if missing: print(f"  Warn: Missing keys loading refiner: {missing}")
        if unexpected: print(f"  Warn: Unexpected keys loading refiner: {unexpected}")
        
        print("Refiner model weights loaded successfully.")
        model_to_eval.eval()
        
    except FileNotFoundError:
        print(f"FATAL: Refiner model file not found at {model_load_path}. Exiting.")
        exit(1)
        
    except Exception as e:
        print(f"FATAL: Error loading refiner model weights: {e}. Check architecture compatibility.")
        traceback.print_exc()
        exit(1)

    print(f"Loading REFINER test data from {args.eval_data_json_path}...")
    args_for_test_loader = argparse.Namespace(**vars(args))
    args_for_test_loader.apply_geom_noise = args.eval_with_geom_noise
    args_for_test_loader.label_corruption_frac_graphs = 0.0

    test_loader_refiner, _, _, _ = r_data_utils.load_refiner_data(
        args_for_test_loader, scaler, label_encoder, is_eval=True # Pass scaler
    )
    if not test_loader_refiner:
        print("FATAL: Refiner test data could not be loaded. Exiting.")
        exit(1)

    main_criterion_eval = nn.CrossEntropyLoss(ignore_index=PAD_VALUE_LABELS_REF)
    correctness_criterion_eval = nn.BCEWithLogitsLoss()

    eval_desc = "Evaluating Refiner"
    if args.eval_with_geom_noise: eval_desc += " (with Geom Noise)"
    
    print(f"\n{eval_desc} on Test Set")
    eval_start_time_actual = time.time()
    (test_main_loss, test_main_acc_subset,
     test_corr_loss, test_corr_acc,
     test_comb_loss,
     _, _, 
     test_corr_preds_all, test_corr_targets_all,
     test_overall_refined_preds_all, test_overall_gt_targets_all,
     test_overall_refined_accuracy) = r_utils.evaluate_refiner(
        model=model_to_eval, loader=test_loader_refiner,
        main_criterion=main_criterion_eval, correctness_criterion=correctness_criterion_eval,
        device=DEVICE, args=args_for_test_loader, desc=eval_desc
    )
    eval_duration_actual = time.time() - eval_start_time_actual
    print(f"Evaluation completed in {eval_duration_actual:.2f} seconds.")

    print(f"\nRefiner Test Set Performance Metrics")
    print(f"  Overall Refined Accuracy (Main Head on All Valid Edges): {test_overall_refined_accuracy:.4f}")
    print(f"  Main Head Loss (on subset of originally incorrect): {test_main_loss:.4f}")
    print(f"  Main Head Accuracy (on subset of originally incorrect): {test_main_acc_subset:.4f}")
    print(f"  Correctness Head Loss (on all valid edges): {test_corr_loss:.4f}")
    print(f"  Correctness Head Accuracy (on all valid edges): {test_corr_acc:.4f}")
    print(f"  Combined Loss (for reference): {test_comb_loss:.4f}")

    target_names = label_encoder.classes_.astype(str)

    print("\nRefiner Overall Classification Report (Main Head on All Test Edges):")
    if test_overall_gt_targets_all and test_overall_refined_preds_all:
         print(classification_report(test_overall_gt_targets_all, test_overall_refined_preds_all,
                                     labels=np.arange(len(target_names)), target_names=target_names, zero_division=0))
         r_utils.plot_confusion_matrix_refiner(
             test_overall_gt_targets_all, test_overall_refined_preds_all,
             target_names, eval_plot_path, # Use path determined by noise flag
             title=f'CM - Refiner Overall ({os.path.basename(args.eval_data_json_path)})',
             output_cm_data_path=eval_cm_data_path
         )
    else: print("  No overall predictions to report/plot.")
    
    print("\nRefiner Correctness Head Classification Report:")
    if test_corr_targets_all and test_corr_preds_all:
        corr_target_names = ['Orig_Incorrect', 'Orig_Correct']
        print(classification_report(test_corr_targets_all, test_corr_preds_all,
                                     labels=[0,1], target_names=corr_target_names, zero_division=0))
        
        corr_plot_path = eval_plot_path.replace(".png", "_correctness_head.png") if eval_plot_path else None
        corr_cm_data_path = eval_cm_data_path.replace(".npy", "_correctness_head.npy") if eval_cm_data_path else None
        
        if corr_plot_path:
            r_utils.plot_confusion_matrix_refiner(
                test_corr_targets_all, test_corr_preds_all,
                corr_target_names, corr_plot_path,
                title=f'CM - Refiner Correctness Head ({os.path.basename(args.eval_data_json_path)})',
                output_cm_data_path=corr_cm_data_path,
                is_binary_correctness=True
            )
    else: print("  No correctness predictions to report/plot.")

    script_end_time = time.time()
    total_script_duration = script_end_time - script_start_time
    print(f"\nRefiner Evaluation Script Finished ({total_script_duration:.2f}s)")

if __name__ == "__main__":
    try:
        main_evaluate_refiner()
    except Exception as e:
        print(f"\nREFINER EVALUATION SCRIPT FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        exit(1)
