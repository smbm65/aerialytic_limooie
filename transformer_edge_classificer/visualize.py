# visualize.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
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

import config
import model
import data_utils
from data_utils import apply_online_geometric_noise, build_adjacency_info
from preprocess import calculate_edge_features


matplotlib.use('Agg')


# Constants
COORD_TOLERANCE = config.COORD_TOLERANCE
PAD_VALUE_LABELS = -100
PAD_VALUE_FEATURES = 0.0
LABEL_COLOR_MAP = {'Ridge': 'red', 'Hip': 'cyan', 'Eave': 'green', 'Rack': 'purple','Flashing': 'lime', 'Valley': 'magenta','Unknown': 'grey'}
DEFAULT_PLOT_COLOR = 'black'
START_POINT_COLOR = 'blue'
END_POINT_COLOR = 'black'
MARKER_SIZE = 10
MARKER_ZORDER = 5
MARKER_OFFSET_DISTANCE = 0.03
MIN_LENGTH_FOR_OFFSET = MARKER_OFFSET_DISTANCE * 2.1

# Main Visualization Logic
def main():
    script_start_time = time.time()
    args = config.setup_arg_parser()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visualization Script")
    print(f"Using device: {DEVICE}")

    print("\nLoading scaler and encoder...")
    try:
        scaler, label_encoder, num_classes = data_utils.load_scaler_encoder(
            args.viz_scaler_path, args.viz_encoder_path
        )
        class_names = label_encoder.classes_.astype(str)
        print(f"  Scaler: {args.viz_scaler_path}")
        print(f"  Encoder: {args.viz_encoder_path}")
        print(f"  Classes: {class_names}")
    except Exception as e:
        print(f"FATAL: Error loading utilities: {e}.")
        exit(1)

    print(f"\nLoading model from {args.viz_model_path}...")
    input_dim = len(config.FEATURE_COLS)
    print(f"Initializing model architecture (Input Dim: {input_dim}, d={args.d_model}, h={args.nhead}, l={args.num_layers}, k={args.k_nearest})...")
    try:
        viz_model = model.TransformerEdgeClassifier(
            input_dim=input_dim, d_model=args.d_model, nhead=args.nhead,
            num_encoder_layers=args.num_layers, dim_feedforward=args.dim_ff,
            num_classes=num_classes,
            dropout=args.dropout,
            norm_first=args.norm_first,
            k_nearest=args.k_nearest,
            mlp_head_dims=args.mlp_head_dims # Pass MLP config
        ).to(DEVICE)
        state_dict = torch.load(args.viz_model_path, map_location=DEVICE)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        missing_keys, unexpected_keys = viz_model.load_state_dict(state_dict, strict=True)
        if missing_keys: print(f"  Warn: Missing keys: {missing_keys}")
        if unexpected_keys: print(f"  Warn: Unexpected keys: {unexpected_keys}")
        viz_model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Error loading model: {e}. Ensure args match saved model.")
        traceback.print_exc()
        exit(1)

    print(f"\nLoading ALL graph data from {args.viz_json_path}...")
    try:
        with open(args.viz_json_path, 'r', encoding='utf-8') as f: all_viz_data = json.load(f)
        print(f"Loaded data for {len(all_viz_data)} graphs.")
    except Exception as e:
        print(f"FATAL ERROR: Error loading viz data: {e}")
        exit(1)

    graph_ids_to_process = []
    if args.viz_input_dir:
        print(f"\nProcessing graphs from directory: {args.viz_input_dir}")
        if not os.path.isdir(args.viz_input_dir):
            print(f"FATAL ERROR: Invalid directory: {args.viz_input_dir}")
            exit(1)
        json_pattern = os.path.join(args.viz_input_dir, "*Graph_Classified.json")
        found_files = glob.glob(json_pattern)
        print(f"Found {len(found_files)} potential JSON files.")
        for f_path in found_files:
            match = re.search(r"Data_(\w+)_Graph_Classified\.json", os.path.basename(f_path))
            if match:
                graph_id = match.group(1)
                if graph_id in all_viz_data: graph_ids_to_process.append(graph_id)
                else: print(f"  Warn: Graph ID '{graph_id}' not in viz JSON.")
            else: print(f"  Warn: Cannot extract ID from: {os.path.basename(f_path)}.")
        if not graph_ids_to_process:
            print(f"FATAL ERROR: No valid graph IDs found.")
            exit(1)
        print(f"Will process {len(graph_ids_to_process)} graphs.")
    elif args.viz_graph_ids:
        print(f"\nProcessing specified graph IDs: {args.viz_graph_ids}")
        for graph_id in args.viz_graph_ids:
            graph_id_str = str(graph_id)
            if graph_id_str in all_viz_data: graph_ids_to_process.append(graph_id_str)
            else: print(f"  Warn: Graph ID '{graph_id_str}' not in viz JSON.")
        if not graph_ids_to_process:
            print(f"FATAL ERROR: None of the specified IDs found.")
            exit(1)
    else:
        print("FATAL ERROR: No graph IDs specified or input directory.")
        exit(1)


    print(f"\nStarting Visualization Loop for {len(graph_ids_to_process)} Graph(s)")
    graphs_processed_count = 0
    for graph_id in graph_ids_to_process:
        print(f"\nProcessing Graph: {graph_id}")
        try:
            original_graph_edges_unfiltered = all_viz_data.get(graph_id)
            if original_graph_edges_unfiltered is None:
                print(f"[{graph_id}] Error: Graph ID not found in visualization data. Skipping.")
                continue
            original_graph_edges_viz = [ edge for edge in original_graph_edges_unfiltered if all(k in edge for k in ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm', 'label_str'])]
            if not original_graph_edges_viz:
                print(f"[{graph_id}] Warn: No valid edge data with required keys. Skip.")
                continue

            noise_suffix = ""
            temp_args = argparse.Namespace(**vars(args))
            apply_noise_this_graph = False
            if args.viz_force_delete_edge:
                temp_args.geom_noise_p_delete_edge = 1.0
                apply_noise_this_graph = True
                noise_suffix += "_delE"
                print(f"    Force edge deletion")
            if args.viz_force_break_edge: 
                temp_args.geom_noise_p_break_edge = 1.0
                apply_noise_this_graph = True
                noise_suffix += "_brk"
                print(f"    Force edge breaking")
            if args.viz_force_angle_noise: 
                temp_args.geom_noise_p_angle_noise = 1.0
                apply_noise_this_graph = True
                noise_suffix += "_ang"
                print(f"    Force angle noise")
            if args.viz_force_length_noise: 
                temp_args.geom_noise_p_length_noise = 1.0
                apply_noise_this_graph = True
                noise_suffix += "_len"
                print(f"    Force length noise")
            if args.viz_force_delete_node: 
                temp_args.geom_noise_p_delete_node = 1.0
                apply_noise_this_graph = True
                noise_suffix += "_delN"
                print(f"    Force node deletion")
            if args.viz_force_subdivide_edge: 
                temp_args.geom_noise_p_subdivide_edge = 1.0
                apply_noise_this_graph = True
                noise_suffix += "_sub"
                print(f"    Force edge subdivision")
            if args.viz_force_coord_noise: 
                temp_args.geom_noise_p_coord_noise = 1.0
                apply_noise_this_graph = True
                noise_suffix += "_coord"
                print(f"    Force coordinate noise")

            if apply_noise_this_graph:
                temp_args.apply_geom_noise = True
                temp_args.geom_noise_global_p = 1.0
            else:
                print(f"[{graph_id}] No noise force flags set.")
                temp_args.apply_geom_noise = False

            adj_info = build_adjacency_info(original_graph_edges_viz)
            graph_edges_for_processing = copy.deepcopy(original_graph_edges_viz)
            noisy_graph_edges = apply_online_geometric_noise(graph_edges_for_processing, adj_info, temp_args)
            if not noisy_graph_edges:
                print(f"[{graph_id}] Warn: All edges removed by noise. Skip plot.")
                continue

            print(f"[{graph_id}] Recalculating features from {'noisy' if temp_args.apply_geom_noise else 'original'} geometry...")
            features_for_model_list, processed_labels_list = [], []
            num_edges_after_noise = len(noisy_graph_edges)
            for edge_dict in noisy_graph_edges:
                try:
                    if not all(k in edge_dict for k in ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm', 'label_str']): continue
                    features = calculate_edge_features(edge_dict['x1_norm'], edge_dict['y1_norm'], edge_dict['x2_norm'], edge_dict['y2_norm'])
                    features_for_model_list.append(features)
                    processed_labels_list.append(edge_dict['label_str'])
                except Exception as e:
                    print(f"[{graph_id}] Err calc features: {e}. Skip edge.")
                    continue
            if not features_for_model_list:
                print(f"[{graph_id}] Err: No features generated. Skip.")
                continue

            df_features_for_model = pd.DataFrame(features_for_model_list)
            try:
                for col in config.FEATURE_COLS:
                    if col not in df_features_for_model.columns:
                        df_features_for_model[col] = 0.0
                        print(f"[{graph_id}] Warn: Missing feature '{col}' filled with 0.")

                features_ordered = df_features_for_model[config.FEATURE_COLS].values.astype(np.float32)
                if features_ordered.shape[0] == 0:
                    print(f"[{graph_id}] Err: No features after ordering. Skip.")
                    continue
                scaled_features = scaler.transform(features_ordered)
                features_tensor = torch.from_numpy(scaled_features).unsqueeze(0).to(DEVICE)
            except Exception as e:
                print(f"[{graph_id}] Err scaling/tensor: {e}. Skip.")
                continue

            print(f"[{graph_id}] Running model inference...")
            start_inf = time.time()
            predicted_labels_noisy = ['InfError'] * num_edges_after_noise
            try:
                with torch.no_grad():
                    logits_seq = viz_model(src=features_tensor, src_key_padding_mask=None)
                    logits = logits_seq.squeeze(0)
                if logits.shape[0] > 0:
                    predicted_indices = torch.argmax(logits, dim=1).cpu().numpy()
                    predicted_indices = np.clip(predicted_indices, 0, len(label_encoder.classes_) - 1)
                    predicted_labels_noisy = list(label_encoder.inverse_transform(predicted_indices))
                else: predicted_labels_noisy = []
                print(f"[{graph_id}] Inference took {time.time() - start_inf:.2f}s.")
                if len(predicted_labels_noisy) != num_edges_after_noise:
                    print(f"[{graph_id}] Warn: Pred count mismatch.")
                    predicted_labels_noisy = (predicted_labels_noisy + ['PredLenError'] * num_edges_after_noise)[:num_edges_after_noise]
            except Exception as e:
                print(f"[{graph_id}] Err inference: {e}")
                traceback.print_exc()

            df_pred_list = [{'x1_norm_pred': ne.get('x1_norm', np.nan), 'y1_norm_pred': ne.get('y1_norm', np.nan), 'x2_norm_pred': ne.get('x2_norm', np.nan), 'y2_norm_pred': ne.get('y2_norm', np.nan), 'predicted_label_str': predicted_labels_noisy[i] if i < len(predicted_labels_noisy) else 'IndexError', 'gt_label_str_aligned': ne.get('label_str', 'UnknownGT')} for i, ne in enumerate(noisy_graph_edges)]
            df_predictions = pd.DataFrame(df_pred_list)
            df_gt_list = [{'x1_norm_orig': ov.get('x1_norm', np.nan), 'y1_norm_orig': ov.get('y1_norm', np.nan), 'x2_norm_orig': ov.get('x2_norm', np.nan), 'y2_norm_orig': ov.get('y2_norm', np.nan), 'gt_label_str': ov.get('label_str', 'UnknownGT')} for ov in original_graph_edges_viz]
            df_ground_truth = pd.DataFrame(df_gt_list)

            if not df_predictions.empty or not df_ground_truth.empty:
                try:
                    base_pattern = args.viz_output_plot_pattern
                    if '{graph_id}' not in base_pattern:
                        name, ext = os.path.splitext(base_pattern)
                        output_filename_i = f"{name}_{graph_id}{noise_suffix}{ext}"
                    else:
                        temp_name = base_pattern.replace('{graph_id}', str(graph_id))
                        name, ext = os.path.splitext(temp_name)
                        output_filename_i = f"{name}{noise_suffix}{ext}"
                except Exception as e:
                    print(f"[{graph_id}] Err fmt plot path: {e}. Fallback.")
                    output_filename_i = os.path.join(args.plot_dir or ".", f"viz_error_{graph_id}{noise_suffix}.png")
                    os.makedirs(os.path.dirname(output_filename_i), exist_ok=True)

                plot_graph_predictions_separate_geom(
                    df_predictions=df_predictions, df_ground_truth=df_ground_truth,
                    class_names=class_names, output_filename=output_filename_i,
                    graph_id=graph_id, show_endpoints=args.viz_show_endpoints
                )
                graphs_processed_count += 1
            else: print(f"[{graph_id}] Skip plot: empty data.")

        except Exception as graph_loop_error:
            print(f"!!! UNEXPECTED ERROR processing graph {graph_id}: {graph_loop_error} !!!")
            traceback.print_exc()

    script_end_time = time.time()
    total_script_duration = script_end_time - script_start_time
    print(f"\nViz Script Finished ({total_script_duration:.2f}s)")
    print(f"Plots generated for {graphs_processed_count} / {len(graph_ids_to_process)} graphs.")


def plot_graph_predictions_separate_geom(df_predictions, df_ground_truth, class_names, output_filename, graph_id, show_endpoints=False):
    print(f"[{graph_id}] Generating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharex=True, sharey=True)
    fig.suptitle(f'Graph {graph_id}: Predictions (Eval Geom) vs. GT (Original Geom)', fontsize=16)
    min_x, max_x, min_y, max_y = -1.1, 1.1, -1.1, 1.1
    x_margin, y_margin = 0, 0
    if not df_ground_truth.empty:
        try:
            all_x_gt = pd.concat([df_ground_truth['x1_norm_orig'], df_ground_truth['x2_norm_orig']]).dropna()
            all_y_gt = pd.concat([df_ground_truth['y1_norm_orig'], df_ground_truth['y2_norm_orig']]).dropna()
            all_x_pred = pd.concat([df_predictions['x1_norm_pred'], df_predictions['x2_norm_pred']]).dropna()
            all_y_pred = pd.concat([df_predictions['y1_norm_pred'], df_predictions['y2_norm_pred']]).dropna()
            all_x = pd.concat([all_x_gt, all_x_pred]).dropna()
            all_y = pd.concat([all_y_gt, all_y_pred]).dropna()
            if not all_x.empty and not all_y.empty:
                min_x, max_x = all_x.min(), all_x.max()
                min_y, max_y = all_y.min(), all_y.max()
                x_range = max(max_x - min_x, 0.1)
                y_range = max(max_y - min_y, 0.1)
                x_margin = x_range * 0.1
                y_margin = y_range * 0.1
            elif not all_x_gt.empty and not all_y_gt.empty:
                 min_x, max_x = all_x_gt.min(), all_x_gt.max()
                 min_y, max_y = all_y_gt.min(), all_y_gt.max()
                 x_range = max(max_x - min_x, 0.1)
                 y_range = max(max_y - min_y, 0.1)
                 x_margin = x_range * 0.1
                 y_margin = y_range * 0.1

        except Exception: pass

    ax_pred = axes[0]
    ax_pred.set_title(f'Predicted Classes (Evaluation Geometry)')
    if not df_predictions.empty:
        for _, edge in df_predictions.iterrows():
            x1, y1, x2, y2, label = edge['x1_norm_pred'], edge['y1_norm_pred'], edge['x2_norm_pred'], edge['y2_norm_pred'], edge['predicted_label_str']
            if pd.isna(label) or label in ['Error','MappingError','IndexError','InfError','NotProcessed','PredLenError'] or pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2): continue
            clr = LABEL_COLOR_MAP.get(label, DEFAULT_PLOT_COLOR)
            ax_pred.plot([x1, x2], [y1, y2], color=clr, linewidth=1.5)
            if show_endpoints:
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                x1m, y1m, x2m, y2m = x1, y1, x2, y2
                if length >= MIN_LENGTH_FOR_OFFSET:
                    ux, uy = (dx/length, dy/length) if length>0 else (0,0)
                    x1m,y1m=x1+MARKER_OFFSET_DISTANCE*ux,y1+MARKER_OFFSET_DISTANCE*uy
                    x2m,y2m=x2-MARKER_OFFSET_DISTANCE*ux,y2-MARKER_OFFSET_DISTANCE*uy
                ax_pred.scatter(x1m, y1m, color=START_POINT_COLOR, s=MARKER_SIZE, zorder=MARKER_ZORDER, alpha=0.9, edgecolors='none')
                ax_pred.scatter(x2m, y2m, color=END_POINT_COLOR, s=MARKER_SIZE, zorder=MARKER_ZORDER, alpha=0.9, edgecolors='none')

    ax_gt = axes[1]
    ax_gt.set_title(f'Ground Truth Classes (Original Geometry)')
    if not df_ground_truth.empty:
        for _, edge in df_ground_truth.iterrows():
            x1, y1, x2, y2, label = edge['x1_norm_orig'], edge['y1_norm_orig'], edge['x2_norm_orig'], edge['y2_norm_orig'], edge['gt_label_str']
            if pd.isna(label) or label=='UnknownGT' or pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2): continue
            clr = LABEL_COLOR_MAP.get(label, DEFAULT_PLOT_COLOR)
            ax_gt.plot([x1, x2], [y1, y2], color=clr, linewidth=1.5)
            if show_endpoints:
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                x1m, y1m, x2m, y2m = x1, y1, x2, y2
                if length >= MIN_LENGTH_FOR_OFFSET:
                    ux, uy = (dx/length, dy/length) if length>0 else (0,0)
                    x1m,y1m=x1+MARKER_OFFSET_DISTANCE*ux,y1+MARKER_OFFSET_DISTANCE*uy
                    x2m,y2m=x2-MARKER_OFFSET_DISTANCE*ux,y2-MARKER_OFFSET_DISTANCE*uy
                ax_gt.scatter(x1m, y1m, color=START_POINT_COLOR, s=MARKER_SIZE, zorder=MARKER_ZORDER, alpha=0.9, edgecolors='none')
                ax_gt.scatter(x2m, y2m, color=END_POINT_COLOR, s=MARKER_SIZE, zorder=MARKER_ZORDER, alpha=0.9, edgecolors='none')

    for ax in axes:
        ax.set_xlabel('X Norm')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(min_x - x_margin, max_x + x_margin)
        ax.set_ylim(min_y - y_margin, max_y + y_margin)
    axes[0].set_ylabel('Y Norm')

    plt.subplots_adjust(bottom=0.2)
    legend_handles = []
    all_pred_labels = set(df_predictions['predicted_label_str'].dropna().unique()) if not df_predictions.empty else set()
    all_gt_labels = set(df_ground_truth['gt_label_str'].dropna().unique()) if not df_ground_truth.empty else set()
    all_labels = all_pred_labels | all_gt_labels
    invalid_legend_labels = ['DELETED','Error','MappingError','IndexError','InfError','NotProcessed','PredLenError','UnknownGT']
    display_labels_filtered = sorted([lbl for lbl in all_labels if not pd.isna(lbl) and lbl not in invalid_legend_labels])
    for label in display_labels_filtered: legend_handles.append(mlines.Line2D([],[],color=LABEL_COLOR_MAP.get(label,DEFAULT_PLOT_COLOR), linewidth=3, label=label))
    if any(lbl not in LABEL_COLOR_MAP for lbl in display_labels_filtered): legend_handles.append(mlines.Line2D([],[],color=DEFAULT_PLOT_COLOR, linewidth=3, label='Other/Default'))
    if show_endpoints:
        legend_handles.append(mlines.Line2D([],[],color=START_POINT_COLOR, marker='o', linestyle='None', markersize=math.sqrt(MARKER_SIZE)*1.5, label='Start'))
        legend_handles.append(mlines.Line2D([],[],color=END_POINT_COLOR, marker='o', linestyle='None', markersize=math.sqrt(MARKER_SIZE)*1.5, label='End'))
    if legend_handles: fig.legend(handles=legend_handles, loc='lower center', ncol=min(len(legend_handles), 8), bbox_to_anchor=(0.5, 0.05))

    fig_to_close = fig
    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        print(f"[{graph_id}] Saving plot to {output_filename}...")
        plt.savefig(output_filename, bbox_inches='tight', dpi=150)
        print(f"[{graph_id}] Plot saved.")
    except Exception as e:
        print(f"[{graph_id}] !!! Error saving plot: {e} !!!")
        traceback.print_exc()
    finally:
        if fig_to_close is not None and plt.fignum_exists(fig_to_close.number): plt.close(fig_to_close)


if __name__ == "__main__":
    main()
