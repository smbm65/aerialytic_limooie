# refiner_main.py
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

import refiner_config as r_config
import refiner_model as r_model
import refiner_data_utils as r_data_utils
import refiner_utils as r_utils
import refiner_trainer as r_trainer


PAD_VALUE_LABELS_REF = r_data_utils.PAD_VALUE_LABELS_REF


def main():
    script_start_time = time.time()
    args = r_config.setup_arg_parser()
    r_config.args_parsed = args 
    r_config.ensure_dirs(args)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Refiner Model Training/Evaluation Script")
    print(f"Using device: {DEVICE}")
    print(f"Refiner HParam String: {args.hparam_str}")
    print(f"Refiner Input Feature Mode: {args.refiner_input_feature_mode}")
    if args.apply_geom_noise: print("Geometric Noise for Refiner: ENABLED during training")
    else: print("Geometric Noise for Refiner: DISABLED during training")
    if args.label_corruption_frac_graphs > 0: print(f"Label Corruption: ENABLED (Graph Frac: {args.label_corruption_frac_graphs}, Edge Frac: {args.label_corruption_frac_edges})")
    else: print("Label Corruption: DISABLED")

    try:
        scaler, label_encoder, num_classes = r_data_utils.load_refiner_scaler_encoder(args)
        if num_classes <= 0: raise ValueError("Number of classes from encoder is not positive.")
    except Exception as e:
        print(f"FATAL: Could not load refiner scaler/encoder. Error: {e}. Run refiner_preprocess.py first.")
        traceback.print_exc()
        exit(1)

    current_input_dim = len(r_config.GEOM_FEATURE_COLS)
    if args.refiner_input_feature_mode == "probs_and_onehot":
        current_input_dim += len(r_config.PROB_FEATURE_COLS) + len(r_config.ONEHOT_FEATURE_COLS)
    elif args.refiner_input_feature_mode == "probs_only":
        current_input_dim += len(r_config.PROB_FEATURE_COLS)
    elif args.refiner_input_feature_mode == "onehot_only":
        current_input_dim += len(r_config.ONEHOT_FEATURE_COLS)
    print(f"Refiner Model Input Dimension (selected mode): {current_input_dim} (Scaled Geom + {args.refiner_input_feature_mode})")

    print("\n Loading Refiner DataLoaders ")
    try:
        train_loader, val_loader, class_weights_main_head, correctness_pos_weight = r_data_utils.load_refiner_data(
            args, scaler, label_encoder, is_eval=False
        )
        if train_loader is None and not args.eval_model_path : 
             print("FATAL: Refiner training data loader failed.")
             exit(1)
        if class_weights_main_head is not None:
            print(f"Refiner main head class weights loaded: {class_weights_main_head.cpu().numpy().round(4)}")
        if correctness_pos_weight is not None:
            print(f"Refiner correctness head pos_weight loaded: {correctness_pos_weight.item():.4f}")
        else:
            print(f"Refiner correctness head loss is un-weighted.")
    except Exception as e:
        print(f"FATAL: Could not load refiner train/val data. Error: {e}")
        traceback.print_exc()
        exit(1)

    print(f"\nInitializing RefinerTransformerEdgeClassifier (Input Dim: {current_input_dim})...")
    refiner_model_instance = r_model.RefinerTransformerEdgeClassifier(
        input_dim=current_input_dim,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_layers, dim_feedforward=args.dim_ff,
        num_classes=num_classes, dropout=args.dropout,
        norm_first=args.norm_first, k_nearest=args.k_nearest,
        knn_distance_threshold=args.knn_distance_threshold,
        refiner_main_head_dims=args.refiner_main_head_dims,
        refiner_correctness_head_dims=args.refiner_correctness_head_dims
    ).to(DEVICE)
    print(f"Refiner Model Trainable Params: {sum(p.numel() for p in refiner_model_instance.parameters() if p.requires_grad):,}")

    main_criterion = nn.CrossEntropyLoss(weight=class_weights_main_head.to(DEVICE) if class_weights_main_head is not None else None,
                                         ignore_index=PAD_VALUE_LABELS_REF)
    
    correctness_criterion = nn.BCEWithLogitsLoss(
        pos_weight=correctness_pos_weight.to(DEVICE) if correctness_pos_weight is not None else None
    )

    optimizer = optim.Adam(refiner_model_instance.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience)

    history = None
    if train_loader:
        history = r_trainer.train_refiner(
            args=args, model=refiner_model_instance,
            train_loader=train_loader, val_loader=val_loader,
            main_criterion=main_criterion, correctness_criterion=correctness_criterion,
            optimizer=optimizer, scheduler=scheduler, device=DEVICE,
            label_encoder=label_encoder
        )
    else:
        print("\n Skipping Refiner Training (no train_loader) ")

    if history and history.get('train_main_loss'):
        print("\n Plotting Refiner Training Convergence ")
        r_utils.plot_convergence_refiner(
            history=history, output_path=args.plot_path_convergence,
            title=f"Refiner Convergence ({args.hparam_str})"
        )
    else: print("\n Skipping refiner convergence plotting (no history) ")

    print("\n Final Refiner Evaluation on Test Set ")
    model_eval_path = args.eval_model_path if args.eval_model_path else args.best_model_path
    print(f"Loading best refiner model from {model_eval_path}...")
    model_to_eval = None
    try:
        best_refiner_model_arch = r_model.RefinerTransformerEdgeClassifier(
            input_dim=current_input_dim, d_model=args.d_model, nhead=args.nhead,
            num_encoder_layers=args.num_layers, dim_feedforward=args.dim_ff,
            num_classes=num_classes, dropout=args.dropout, norm_first=args.norm_first,
            k_nearest=args.k_nearest, knn_distance_threshold=args.knn_distance_threshold,
            refiner_main_head_dims=args.refiner_main_head_dims,
            refiner_correctness_head_dims=args.refiner_correctness_head_dims
        ).to(DEVICE)
        
        if not os.path.exists(model_eval_path):
            print(f"Warning: Model for eval '{model_eval_path}' not found.")
            if os.path.exists(args.latest_ckpt_path):
                 print(f"Attempting to load from latest checkpoint '{args.latest_ckpt_path}' instead.")
                 model_eval_path = args.latest_ckpt_path
            else:
                raise FileNotFoundError(f"Neither best model nor latest checkpoint found.")

        state_dict_payload = torch.load(model_eval_path, map_location=DEVICE)
        state_dict = state_dict_payload['model_state_dict'] if isinstance(state_dict_payload, dict) and 'model_state_dict' in state_dict_payload else state_dict_payload
        missing, unexpected = best_refiner_model_arch.load_state_dict(state_dict, strict=False)
        if missing: print(f"  Warn: Missing keys loading best refiner model: {missing}")
        if unexpected: print(f"  Warn: Unexpected keys loading best refiner model: {unexpected}")
        model_to_eval = best_refiner_model_arch
        model_to_eval.eval()
    except FileNotFoundError:
        print(f"Warning: Model for eval '{model_eval_path}' not found. Using final model from training if available.")
        model_to_eval = refiner_model_instance
        if model_to_eval: model_to_eval.eval()
    except Exception as e:
        print(f"Warning: Error loading model for eval ({e}). Using final model from training if available.")
        traceback.print_exc()
        model_to_eval = refiner_model_instance
        if model_to_eval: model_to_eval.eval()

    if not model_to_eval:
        print("FATAL: No model available for final evaluation.")
        exit(1)

    args_eval_test = argparse.Namespace(**vars(args))
    args_eval_test.apply_geom_noise = args.eval_with_geom_noise
    args_eval_test.label_corruption_frac_graphs = 0.0

    test_loader_refiner, _, _, _ = r_data_utils.load_refiner_data(
        args_eval_test, scaler, label_encoder, is_eval=True
    )

    if test_loader_refiner:
        eval_desc = "Testing Refiner"
        if args.eval_with_geom_noise: eval_desc += " (with Geom Noise)"
        
        (test_main_loss, test_main_acc_subset, test_corr_loss, test_corr_acc, test_comb_loss,
         test_main_preds_subset, test_main_targets_subset, test_corr_preds_all, test_corr_targets_all,
         test_overall_refined_preds_all, test_overall_gt_targets_all, test_overall_refined_accuracy
        ) = r_utils.evaluate_refiner(
            model=model_to_eval, loader=test_loader_refiner,
            main_criterion=main_criterion, correctness_criterion=correctness_criterion,
            device=DEVICE, args=args_eval_test, desc=eval_desc
        )

        print(f"\nRefiner Test Set Performance")
        print(f"  Overall Refined Accuracy (Main Head on All Valid Edges): {test_overall_refined_accuracy:.4f}")
        print(f"  Main Head Loss (on subset of originally incorrect): {test_main_loss:.4f}")
        print(f"  Main Head Accuracy (on subset of originally incorrect): {test_main_acc_subset:.4f}")
        print(f"  Correctness Head Loss: {test_corr_loss:.4f}")
        print(f"  Correctness Head Accuracy: {test_corr_acc:.4f}")
        print(f"  Combined Loss (for reference): {test_comb_loss:.4f}")

        target_names = label_encoder.classes_.astype(str)
        plot_path_overall_cm = args.eval_plot_path
        cm_data_path_overall = args.eval_cm_data_path
        if args.eval_with_geom_noise:
            plot_path_overall_cm = r_config.get_noisy_eval_plot_path(args, for_refiner=True)
            cm_data_path_overall = cm_data_path_overall.replace(".npy", "_geom_noisy.npy")

        print("\nRefiner Overall Classification Report (Main Head on All Test Edges):")
        if test_overall_gt_targets_all and test_overall_refined_preds_all: 
             print(classification_report(test_overall_gt_targets_all, test_overall_refined_preds_all, labels=np.arange(len(target_names)), target_names=target_names, zero_division=0))
             r_utils.plot_confusion_matrix_refiner(test_overall_gt_targets_all, test_overall_refined_preds_all, target_names, plot_path_overall_cm, title=f'CM - Refiner Overall Output ({args.hparam_str})', output_cm_data_path=cm_data_path_overall)
        else: print("  No overall predictions to report/plot.")
        
        print("\nRefiner Correctness Head Classification Report:")
        if test_corr_targets_all and test_corr_preds_all:
            corr_target_names = ['Orig_Incorrect', 'Orig_Correct']
            print(classification_report(test_corr_targets_all, test_corr_preds_all, labels=[0,1], target_names=corr_target_names, zero_division=0))
            plot_path_corr_cm = plot_path_overall_cm.replace(".png", "_correctness_head.png")
            cm_data_path_corr = cm_data_path_overall.replace(".npy", "_correctness_head.npy") if cm_data_path_overall else None
            r_utils.plot_confusion_matrix_refiner(test_corr_targets_all, test_corr_preds_all, corr_target_names, plot_path_corr_cm, title=f'CM - Refiner Correctness Head ({args.hparam_str})', output_cm_data_path=cm_data_path_corr, is_binary_correctness=True)
        else: print("  No correctness predictions to report/plot.")
    else:
        print("Refiner test data not loaded. Skipping final eval.")

    script_end_time = time.time()
    total_duration = script_end_time - script_start_time
    print(f"\nRefiner Main Script Finished ({total_duration:.2f}s)")

if __name__ == "__main__":
    main()
