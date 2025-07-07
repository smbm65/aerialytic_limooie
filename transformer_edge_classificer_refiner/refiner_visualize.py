# refiner_visualize.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import math
import time
import os
import json
import glob
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
import traceback
import random
import copy
import argparse

import refiner_config as r_config
import refiner_model as r_model
import refiner_data_utils as r_data_utils


matplotlib.use('Agg') 


LABEL_COLOR_MAP_REF = {'Ridge': 'red', 'Hip': 'cyan', 'Eave': 'green', 'Rack': 'purple','Flashing': 'lime', 'Valley': 'magenta','Unknown': 'grey', 'Error': 'black', 'PadErrorViz': 'orange', 'ErrorViz': 'black', 'UnknownGT': 'grey'}
DEFAULT_PLOT_COLOR_REF = 'darkgoldenrod'
START_POINT_COLOR_REF = 'blue'
END_POINT_COLOR_REF = 'black'
MARKER_SIZE_REF = 8
MARKER_ZORDER_REF = 5

PAD_VALUE_LABELS_REF = r_data_utils.PAD_VALUE_LABELS_REF


def plot_refiner_outputs_triple(
    df_ground_truth, df_input_to_refiner, df_refiner_predictions,
    class_names, output_filename, graph_id, show_endpoints=False
    ):
    print(f"[{graph_id}] Generating triple plot for refiner visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(27, 9), sharex=True, sharey=True)
    fig.suptitle(f'Graph {graph_id}: GT vs. Input to Refiner vs. Refiner Output (Effective)', fontsize=16)

    min_x_overall, max_x_overall, min_y_overall, max_y_overall = float('inf'), float('-inf'), float('inf'), float('-inf')
    
    def update_bounds(df, x1c, y1c, x2c, y2c):
        nonlocal min_x_overall, max_x_overall, min_y_overall, max_y_overall
        if not df.empty and all(c in df.columns for c in [x1c,y1c,x2c,y2c]):
            # Filter out NaNs before min/max
            x1_valid = df[x1c].dropna()
            y1_valid = df[y1c].dropna()
            x2_valid = df[x2c].dropna()
            y2_valid = df[y2c].dropna()
            
            current_xs = pd.concat([x1_valid, x2_valid])
            current_ys = pd.concat([y1_valid, y2_valid])

            if not current_xs.empty:
                min_x_overall = min(min_x_overall, current_xs.min())
                max_x_overall = max(max_x_overall, current_xs.max())
            if not current_ys.empty:
                min_y_overall = min(min_y_overall, current_ys.min())
                max_y_overall = max(max_y_overall, current_ys.max())

    update_bounds(df_ground_truth, 'x1_norm_gt', 'y1_norm_gt', 'x2_norm_gt', 'y2_norm_gt')
    update_bounds(df_input_to_refiner, 'x1_norm_in', 'y1_norm_in', 'x2_norm_in', 'y2_norm_in')
    update_bounds(df_refiner_predictions, 'x1_norm_pred', 'y1_norm_pred', 'x2_norm_pred', 'y2_norm_pred')

    # Default if no valid coords
    if min_x_overall == float('inf'): min_x_overall, max_x_overall, min_y_overall, max_y_overall = -1.1, 1.1, -1.1, 1.1
    
    x_range, y_range = max(max_x_overall - min_x_overall, 0.1), max(max_y_overall - min_y_overall, 0.1)
    x_margin, y_margin = x_range * 0.05, y_range * 0.05

    plot_titles = ['Ground Truth (Original Geom)', 'Input to Refiner (Orig. Preds)', 'Effective Refiner Output']
    dataframes = [df_ground_truth, df_input_to_refiner, df_refiner_predictions]
    cols_map = [
        {'x1':'x1_norm_gt', 'y1':'y1_norm_gt', 'x2':'x2_norm_gt', 'y2':'y2_norm_gt', 'label':'gt_label_str'},
        {'x1':'x1_norm_in', 'y1':'y1_norm_in', 'x2':'x2_norm_in', 'y2':'y2_norm_in', 'label':'input_label_str'},
        {'x1':'x1_norm_pred', 'y1':'y1_norm_pred', 'x2':'x2_norm_pred', 'y2':'y2_norm_pred', 'label':'refined_label_str'}
    ]

    for i, ax in enumerate(axes):
        ax.set_title(plot_titles[i])
        df, cols = dataframes[i], cols_map[i]
        if not df.empty:
            for _, edge in df.iterrows():
                x1,y1,x2,y2 = edge.get(cols['x1']), edge.get(cols['y1']), edge.get(cols['x2']), edge.get(cols['y2'])
                label_str = edge.get(cols['label'], 'Unknown')
                if any(pd.isna([x1,y1,x2,y2])): continue
                color = LABEL_COLOR_MAP_REF.get(label_str, DEFAULT_PLOT_COLOR_REF)
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=0.8)
                if show_endpoints: 
                    ax.scatter([x1, x2], [y1, y2], color=[START_POINT_COLOR_REF, END_POINT_COLOR_REF], s=MARKER_SIZE_REF, zorder=MARKER_ZORDER_REF, alpha=0.7)
        ax.set_xlabel('X Norm')
        ax.set_ylabel('Y Norm' if i==0 else '')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(min_x_overall - x_margin, max_x_overall + x_margin)
        ax.set_ylim(min_y_overall - y_margin, max_y_overall + y_margin)

    unique_labels_all_plots = set()
    
    if not df_ground_truth.empty and cols_map[0]['label'] in df_ground_truth.columns:
        unique_labels_all_plots.update(df_ground_truth[cols_map[0]['label']].dropna().unique())
    if not df_input_to_refiner.empty and cols_map[1]['label'] in df_input_to_refiner.columns:
        unique_labels_all_plots.update(df_input_to_refiner[cols_map[1]['label']].dropna().unique())
    if not df_refiner_predictions.empty and cols_map[2]['label'] in df_refiner_predictions.columns:
        unique_labels_all_plots.update(df_refiner_predictions[cols_map[2]['label']].dropna().unique())
    
    legend_handles = [mlines.Line2D([],[],color=LABEL_COLOR_MAP_REF.get(lbl, DEFAULT_PLOT_COLOR_REF), lw=3, label=lbl) for lbl in sorted(list(unique_labels_all_plots)) if lbl not in ['ErrorViz', 'PadErrorViz']]
    
    if show_endpoints:
        legend_handles.append(mlines.Line2D([],[],color=START_POINT_COLOR_REF, marker='o', ls='None', label='Start'))
        legend_handles.append(mlines.Line2D([],[],color=END_POINT_COLOR_REF, marker='o', ls='None', label='End'))
    
    if legend_handles: 
        fig.legend(handles=legend_handles, loc='lower center', ncol=min(len(legend_handles), 8), bbox_to_anchor=(0.5, 0.01))
    
    plt.subplots_adjust(bottom=0.15 if legend_handles else 0.05, top=0.92)

    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir : os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_filename, bbox_inches='tight', dpi=150)
        print(f"[{graph_id}] Triple plot saved to {output_filename}.")
    except Exception as e: 
        print(f"[{graph_id}] Error saving triple plot: {e}")
        traceback.print_exc()
    finally: plt.close(fig)


def main_visualize_refiner():
    script_start_time = time.time()
    args = r_config.setup_arg_parser()
    r_config.args_parsed = args
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Refiner Visualization Script\nUsing device: {DEVICE}")

    try:
        scaler, label_encoder, num_classes = r_data_utils.load_refiner_scaler_encoder(args)
        class_names = label_encoder.classes_.astype(str)
    except Exception as e: 
        print(f"FATAL: Error loading utilities: {e}.")
        traceback.print_exc()
        exit(1)

    current_input_dim = len(r_config.GEOM_FEATURE_COLS)
    if args.refiner_input_feature_mode == "probs_and_onehot": current_input_dim += len(r_config.PROB_FEATURE_COLS) + len(r_config.ONEHOT_FEATURE_COLS)
    elif args.refiner_input_feature_mode == "probs_only": current_input_dim += len(r_config.PROB_FEATURE_COLS)
    elif args.refiner_input_feature_mode == "onehot_only": current_input_dim += len(r_config.ONEHOT_FEATURE_COLS)

    viz_model_path = args.viz_model_path
    print(f"\nLoading refiner model from {viz_model_path}...")
    try:
        viz_refiner_model = r_model.RefinerTransformerEdgeClassifier(
            input_dim=current_input_dim, d_model=args.d_model, nhead=args.nhead,
            num_encoder_layers=args.num_layers, dim_feedforward=args.dim_ff,
            num_classes=num_classes, dropout=args.dropout, norm_first=args.norm_first,
            k_nearest=args.k_nearest, knn_distance_threshold=args.knn_distance_threshold
        ).to(DEVICE)
        
        if not os.path.exists(viz_model_path):
            print(f"FATAL ERROR: Model for visualization not found at {viz_model_path}")
            exit(1)
            
        state_dict_payload = torch.load(viz_model_path, map_location=DEVICE)
        if isinstance(state_dict_payload, dict) and 'model_state_dict' in state_dict_payload: 
            state_dict = state_dict_payload['model_state_dict']
        else:
            state_dict = state_dict_payload # Assumed raw state_dict

        missing, unexpected = viz_refiner_model.load_state_dict(state_dict, strict=False)
        
        if missing: print(f"  Warn (Viz Load): Missing keys: {missing}")
        if unexpected: print(f"  Warn (Viz Load): Unexpected keys: {unexpected}")
        
        viz_refiner_model.eval()
        print("Refiner model loaded for visualization.")

    except Exception as e: 
        print(f"FATAL ERROR: Loading refiner model for viz: {e}.")
        traceback.print_exc()
        exit(1)

    print(f"\nLoading graph ID list from manifest: {args.viz_json_path}")
    all_graph_ids_from_manifest = []
    try:
        with open(args.viz_json_path, 'r', encoding='utf-8') as f_manifest:
            all_graph_ids_from_manifest = json.load(f_manifest)
        print(f"Loaded {len(all_graph_ids_from_manifest)} graph IDs from manifest for potential visualization.")
    except Exception as e:
        print(f"FATAL ERROR: Loading refiner visualization manifest JSON '{args.viz_json_path}': {e}")
        exit(1)

    graph_ids_to_process = args.viz_graph_ids if args.viz_graph_ids else all_graph_ids_from_manifest
    
    if not graph_ids_to_process: 
        print("No graph IDs to visualize. Exiting.")
        exit(0)

    print(f"Will attempt to visualize {len(graph_ids_to_process)} graphs.")

    graphs_plotted_count = 0
    for graph_id_str in graph_ids_to_process:
        print(f"\nVisualizing Graph: {graph_id_str}")
        
        individual_graph_json_path = os.path.join(args.dataset_dir, "graph_data", f"{graph_id_str}.json")
        graph_edge_data_from_json = None
        try:
            with open(individual_graph_json_path, 'r', encoding='utf-8') as f_graph:
                graph_edge_data_from_json = json.load(f_graph) # List of edge dicts
        except FileNotFoundError:
            print(f"  [{graph_id_str}] Individual graph JSON not found at {individual_graph_json_path}. Skipping.")
            continue
        except Exception as e_load_indiv:
            print(f"  [{graph_id_str}] Error loading individual graph JSON {individual_graph_json_path}: {e_load_indiv}. Skipping.")
            continue
            
        if not graph_edge_data_from_json: 
            print(f"  [{graph_id_str}] Data is empty after loading. Skipping.")
            continue

        args_for_collate_viz = argparse.Namespace(**vars(args))
        args_for_collate_viz.apply_geom_noise = args.viz_force_coord_noise
        args_for_collate_viz.label_corruption_frac_graphs = 1.0 if args.viz_corrupt_labels else 0.0
        args_for_collate_viz.label_corruption_frac_edges = args.label_corruption_frac_edges
        
        noise_suffix = ""
        if args_for_collate_viz.apply_geom_noise: noise_suffix += "_geomNoisy"
        if args_for_collate_viz.label_corruption_frac_graphs > 0 : noise_suffix += "_lblCorr"
        
        try:
            features_tsr, gt_labels_tsr, orig_pred_idx_tsr, correctness_tgt_tsr, attn_mask_tsr = \
                r_data_utils.refiner_collate_fn([graph_edge_data_from_json], args_for_collate_viz, scaler, num_classes, is_eval=True)
        except Exception as e_collate: 
            print(f"  [{graph_id_str}] Error during collate_fn for visualization: {e_collate}. Skipping.")
            traceback.print_exc()
            continue

        valid_len = (~attn_mask_tsr[0]).sum().item() # Number of actual edges in this graph
        if valid_len == 0:
            print(f"  [{graph_id_str}] No valid edges after collate. Skipping.")
            continue

        with torch.no_grad():
            main_logits_viz, correctness_logits_viz = viz_refiner_model(features_tsr.to(DEVICE), attn_mask_tsr.to(DEVICE))
        
        coords_x1 = [edge.get('x1_norm', np.nan) for edge in graph_edge_data_from_json][:valid_len]
        coords_y1 = [edge.get('y1_norm', np.nan) for edge in graph_edge_data_from_json][:valid_len]
        coords_x2 = [edge.get('x2_norm', np.nan) for edge in graph_edge_data_from_json][:valid_len]
        coords_y2 = [edge.get('y2_norm', np.nan) for edge in graph_edge_data_from_json][:valid_len]

        # Ground Truth Labels (from the original data)
        gt_labels_str_viz = [edge.get(r_config.LABEL_COL_STR_REF, 'UnknownGT') for edge in graph_edge_data_from_json][:valid_len]
        
        input_to_refiner_pred_indices_viz = orig_pred_idx_tsr[0, :valid_len].cpu().numpy()
        
        # Handle padding for inverse_transform
        valid_input_indices_mask = (input_to_refiner_pred_indices_viz != PAD_VALUE_LABELS_REF) & \
                                   (input_to_refiner_pred_indices_viz < num_classes) & \
                                   (input_to_refiner_pred_indices_viz >=0)

        input_to_refiner_labels_viz_list = ['PadErrorViz'] * valid_len 
        if np.any(valid_input_indices_mask):
            valid_labels = label_encoder.inverse_transform(input_to_refiner_pred_indices_viz[valid_input_indices_mask])
            count = 0
            for i in range(valid_len):
                if valid_input_indices_mask[i]:
                    input_to_refiner_labels_viz_list[i] = valid_labels[count]
                    count += 1
        
        # Effective Refiner Output (combining main head and correctness head predictions)
        raw_main_head_pred_indices = torch.argmax(main_logits_viz[0, :valid_len, :], dim=1).cpu().numpy()
        # Correctness prediction: 0 means original was WRONG, 1 means original was CORRECT
        correctness_pred_binary = (torch.sigmoid(correctness_logits_viz[0, :valid_len, :].squeeze(-1)) > 0.5).long().cpu().numpy()
        
        final_effective_pred_indices = np.zeros_like(raw_main_head_pred_indices) # Default to class 0
        for i in range(valid_len):
            original_idx_for_this_edge = input_to_refiner_pred_indices_viz[i]
            main_head_idx = raw_main_head_pred_indices[i]
            
            # Check if original_idx is a valid class index before using it
            is_original_idx_valid_class = (original_idx_for_this_edge != PAD_VALUE_LABELS_REF) and \
                                          (0 <= original_idx_for_this_edge < num_classes)

            if correctness_pred_binary[i] == 0: # Predicts original was WRONG (or original was invalid padding) -> use main_head
                final_effective_pred_indices[i] = main_head_idx
            elif is_original_idx_valid_class : # Predicts original was CORRECT and original_idx is a valid class
                final_effective_pred_indices[i] = original_idx_for_this_edge
            else: # Predicts original was CORRECT, but original_idx was padding/invalid -> default to main_head
                final_effective_pred_indices[i] = main_head_idx


        valid_refined_indices_mask = (final_effective_pred_indices != PAD_VALUE_LABELS_REF) & \
                                     (final_effective_pred_indices < num_classes) & \
                                     (final_effective_pred_indices >=0)
        final_effective_labels_viz_list = ['PadErrorViz'] * valid_len
        if np.any(valid_refined_indices_mask):
            valid_refined_labels = label_encoder.inverse_transform(final_effective_pred_indices[valid_refined_indices_mask])
            count = 0
            for i in range(valid_len):
                if valid_refined_indices_mask[i]:
                    final_effective_labels_viz_list[i] = valid_refined_labels[count]
                    count +=1

        df_gt = pd.DataFrame({'x1_norm_gt': coords_x1, 'y1_norm_gt': coords_y1, 'x2_norm_gt': coords_x2, 'y2_norm_gt': coords_y2, 'gt_label_str': gt_labels_str_viz})
        df_in = pd.DataFrame({'x1_norm_in': coords_x1, 'y1_norm_in': coords_y1, 'x2_norm_in': coords_x2, 'y2_norm_in': coords_y2, 'input_label_str': input_to_refiner_labels_viz_list})
        df_pred = pd.DataFrame({'x1_norm_pred': coords_x1, 'y1_norm_pred': coords_y1, 'x2_norm_pred': coords_x2, 'y2_norm_pred': coords_y2, 'refined_label_str': final_effective_labels_viz_list})

        output_filename = args.viz_output_plot_pattern.format(graph_id=graph_id_str)
        if noise_suffix:
            name, ext = os.path.splitext(output_filename)
            output_filename = f"{name}{noise_suffix}{ext}"

        plot_refiner_outputs_triple(df_gt, df_in, df_pred, class_names, output_filename, graph_id_str, args.viz_show_endpoints)
        graphs_plotted_count +=1

    total_duration = time.time() - script_start_time
    print(f"\nRefiner Visualization Script Finished ({total_duration:.2f}s)")
    print(f"Plots generated for {graphs_plotted_count}/{len(graph_ids_to_process)} graphs.")

if __name__ == "__main__":
    try: main_visualize_refiner()
    except Exception as e: 
        print(f"\nSCRIPT FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        exit(1)
