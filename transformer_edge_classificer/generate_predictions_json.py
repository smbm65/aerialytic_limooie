# generate_predictions_json.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import math
import time
import os
import json
import glob
import re
import sys
import traceback
import random
import copy
import argparse
from collections import defaultdict
from tqdm import tqdm

# Import project modules
import config
import model
from data_utils import apply_online_geometric_noise, build_adjacency_info, load_scaler_encoder
from preprocess import calculate_edge_features

def parse_raw_graph_data(graph_data_raw_json_list):
    """
    Parses the raw nested list format from *_Graph_Classified.json
    into a list of dictionaries with raw coordinates and labels.
    """
    parsed_edges = []
    for item_idx, item in enumerate(graph_data_raw_json_list):
        if not (isinstance(item, list) and len(item) >= 2 and
                isinstance(item[0], list) and len(item[0]) == 2 and
                isinstance(item[0][0], list) and len(item[0][0]) == 2 and
                isinstance(item[0][1], list) and len(item[0][1]) == 2 and
                isinstance(item[1], str)):
            continue
        
        coords_raw_list, label_str = item[0], item[1]
        p1_raw_list, p2_raw_list = coords_raw_list[0], coords_raw_list[1]
        
        try:
            x_1, y_1 = float(p1_raw_list[0]), float(p1_raw_list[1])
            x_2, y_2 = float(p2_raw_list[0]), float(p2_raw_list[1])

            if abs(x_1 - x_2) < config.COORD_TOLERANCE and abs(y_1 - y_2) < config.COORD_TOLERANCE:
                continue

            x1_canon_raw, y1_canon_raw, x2_canon_raw, y2_canon_raw = x_1, y_1, x_2, y_2
            swap_points = False
            if x1_canon_raw > x2_canon_raw + config.COORD_TOLERANCE: swap_points = True
            elif abs(x1_canon_raw - x2_canon_raw) < config.COORD_TOLERANCE and \
                 y1_canon_raw > y2_canon_raw + config.COORD_TOLERANCE: swap_points = True
            
            if swap_points:
                x1_canon_raw, y1_canon_raw, x2_canon_raw, y2_canon_raw = \
                    x2_canon_raw, y2_canon_raw, x1_canon_raw, y1_canon_raw

            parsed_edges.append({
                'x1_raw': x1_canon_raw, 'y1_raw': y1_canon_raw,
                'x2_raw': x2_canon_raw, 'y2_raw': y2_canon_raw,
                'label_str': label_str
            })
        except (ValueError, TypeError) as e:
            continue
    return parsed_edges


def normalize_graph_edges_per_graph(raw_edge_list, norm_method="bbox_max_dim"):
    """
    Normalizes a list of raw edge dictionaries FOR A SINGLE GRAPH.
    Returns a list of dictionaries with 'x1_norm', 'y1_norm', etc. and original 'label_str'.
    """
    if not raw_edge_list:
        return []

    all_coords_raw = []
    for edge in raw_edge_list:
        all_coords_raw.append((edge['x1_raw'], edge['y1_raw']))
        all_coords_raw.append((edge['x2_raw'], edge['y2_raw']))
    
    if not all_coords_raw:
        return []

    nodes_array = np.array(all_coords_raw)
    xmin, ymin = nodes_array.min(axis=0)
    xmax, ymax = nodes_array.max(axis=0)
    
    width = xmax - xmin
    height = ymax - ymin

    if norm_method == "bbox_max_dim":
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        scale_factor = max(width / 2.0, height / 2.0, 1e-9) 
        offset_x, offset_y = center_x, center_y
    elif norm_method == "normlen":
        scale_factor = 1.0
        offset_x = 0.0
        offset_y = 0.0
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

    normalized_edges = []
    for edge_raw_dict in raw_edge_list:
        x1n = (edge_raw_dict['x1_raw'] - offset_x) / scale_factor
        y1n = (edge_raw_dict['y1_raw'] - offset_y) / scale_factor
        x2n = (edge_raw_dict['x2_raw'] - offset_x) / scale_factor
        y2n = (edge_raw_dict['y2_raw'] - offset_y) / scale_factor
        
        normalized_edges.append({
            'x1_norm': x1n, 'y1_norm': y1n,
            'x2_norm': x2n, 'y2_norm': y2n,
            'label_str': edge_raw_dict['label_str']
        })
    return normalized_edges


def process_and_save_batch(batch_features_tensors_list, batch_meta_info_list, pred_model, device, feature_dim):
    """
    Processes a batch of graph features, gets predictions, and saves them.
    """
    if not batch_features_tensors_list:
        return

    num_graphs_in_batch = len(batch_features_tensors_list)
    
    max_seq_len = 0
    for features_tensor in batch_features_tensors_list: 
        if features_tensor.size(0) > max_seq_len:
            max_seq_len = features_tensor.size(0)
    
    if max_seq_len == 0:
        for meta_info in batch_meta_info_list:
             tqdm.write(f"    Graph {meta_info['graph_id']}: Was empty or yielded no features for batching. Skipping save.")
        return

    padded_features_batch = torch.zeros(num_graphs_in_batch, max_seq_len, feature_dim, dtype=torch.float32, device=device)
    src_key_padding_mask_batch = torch.ones(num_graphs_in_batch, max_seq_len, dtype=torch.bool, device=device)

    for i, features_tensor in enumerate(batch_features_tensors_list):
        current_seq_len = features_tensor.size(0)
        if current_seq_len > 0 :
            padded_features_batch[i, :current_seq_len, :] = features_tensor
            src_key_padding_mask_batch[i, :current_seq_len] = False 

    with torch.no_grad():
        logits_batch_seq = pred_model(src=padded_features_batch, src_key_padding_mask=src_key_padding_mask_batch)
        probabilities_batch = F.softmax(logits_batch_seq, dim=2) 
        probabilities_batch_cpu_list = probabilities_batch.cpu().tolist()

    for i in range(num_graphs_in_batch):
        meta_info = batch_meta_info_list[i]
        original_num_edges = meta_info['original_num_edges']
        
        if original_num_edges == 0:
            continue

        graph_probabilities_list = probabilities_batch_cpu_list[i][:original_num_edges]
        geometry_and_labels_for_output = meta_info['geometry_and_labels']

        if len(geometry_and_labels_for_output) != len(graph_probabilities_list):
            tqdm.write(f"    Graph {meta_info['graph_id']}: Mismatch during batch post-processing! "
                       f"Geom/labels count ({len(geometry_and_labels_for_output)}) != "
                       f"Probs count ({len(graph_probabilities_list)}). Original edges: {original_num_edges}. "
                       f"Skipping save for this graph.")
            continue

        output_json_content = []
        for j in range(original_num_edges): 
            geom_label_entry = geometry_and_labels_for_output[j]
            output_json_content.append({
                "x1_norm": geom_label_entry['x1_norm'],
                "y1_norm": geom_label_entry['y1_norm'],
                "x2_norm": geom_label_entry['x2_norm'],
                "y2_norm": geom_label_entry['y2_norm'],
                "label_str": geom_label_entry['label_str'],
                "probabilities": graph_probabilities_list[j]
            })
        
        try:
            with open(meta_info['output_filepath'], 'w', encoding='utf-8') as f_out:
                json.dump(output_json_content, f_out, indent=2)
        except Exception as e_save:
            tqdm.write(f"    Error saving output for graph {meta_info['graph_id']} to {meta_info['output_filepath']}: {e_save}")


def main_generate(args):
    """
    Main function for generating predictions.
    It now directly uses the 'args' object parsed in __main__.
    """
    script_start_time = time.time()
    batch_size = args.batch_size

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Prediction JSON Generation Script (Batch Size: {batch_size}) ---")
    print(f"Using device: {DEVICE}")
    print(f"Model: {args.model_path}")
    print(f"Scaler: {args.scaler_path}")
    print(f"Encoder: {args.encoder_path}")
    print(f"Normalization method for input: {args.norm_method}")

    print("\nLoading scaler and encoder...")
    try:
        scaler, label_encoder, num_classes = load_scaler_encoder(
            args.scaler_path, args.encoder_path
        )
        class_names = label_encoder.classes_.astype(str)
        print(f"  Classes: {class_names}")
    except Exception as e:
        print(f"FATAL: Error loading utilities: {e}. Check paths.")
        traceback.print_exc()
        exit(1)

    print(f"\nLoading model from {args.model_path}...")
    input_dim = len(config.FEATURE_COLS)
    print(f"Initializing model architecture (Input Dim: {input_dim}, d={args.d_model}, h={args.nhead}, l={args.num_layers}, k={args.k_nearest})...")
    try:
        pred_model = model.TransformerEdgeClassifier(
            input_dim=input_dim, d_model=args.d_model, nhead=args.nhead,
            num_encoder_layers=args.num_layers, dim_feedforward=args.dim_ff,
            num_classes=num_classes, dropout=args.dropout,
            norm_first=args.norm_first, k_nearest=args.k_nearest,
            mlp_head_dims=args.mlp_head_dims
        ).to(DEVICE)
        state_dict = torch.load(args.model_path, map_location=DEVICE)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        missing_keys, unexpected_keys = pred_model.load_state_dict(state_dict, strict=True)
        if missing_keys: print(f"  Warning: Missing keys in model state_dict: {missing_keys}")
        if unexpected_keys: print(f"  Warning: Unexpected keys in model state_dict: {unexpected_keys}")
        pred_model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Error loading model: {e}. Ensure CLI args for model structure match SAVED model.")
        traceback.print_exc()
        exit(1)

    dataset_types = {"Train": args.train_input_dir, "Test": args.test_input_dir}
    modes = ["clean", "noisy"]

    for mode in modes:
        if mode == "noisy" and not args.include_noisy:
            print(f"Skipping noisy mode as per --no-noisy flag.")
            continue
        
        args_for_current_mode = copy.deepcopy(args)
        if mode == "noisy":
            args_for_current_mode.apply_geom_noise = True
            # Set global probability to 1.0 to ensure noise is attempted on every graph
            args_for_current_mode.geom_noise_global_p = 1.0 
            print(f"\n--- Generating for mode: NOISY ---")
        else:
            args_for_current_mode.apply_geom_noise = False
            print(f"\n--- Generating for mode: CLEAN ---")

        for ds_type, input_dir_path in dataset_types.items():
            if not input_dir_path or not os.path.isdir(input_dir_path):
                print(f"Warning: Input directory for {ds_type} ('{input_dir_path}') is invalid or not provided. Skipping.")
                continue

            output_dir_path = os.path.join(args.output_base_dir, f"{mode}_output", ds_type.lower())
            os.makedirs(output_dir_path, exist_ok=True)
            print(f"  Processing {ds_type} data from: {input_dir_path}")
            print(f"  Outputting to: {output_dir_path}")

            json_files_to_process = glob.glob(os.path.join(input_dir_path, "*_Graph_Classified.json"))
            if not json_files_to_process:
                print(f"  No '*_Graph_Classified.json' files found in {input_dir_path}. Skipping.")
                continue
            
            print(f"  Found {len(json_files_to_process)} JSON files for {ds_type}.")

            current_batch_features_tensors = []
            current_batch_meta_info = []

            for file_idx, json_filepath in enumerate(tqdm(json_files_to_process, desc=f"Processing {ds_type} ({mode})", unit="file", leave=True)):
                filename = os.path.basename(json_filepath)
                match = re.match(r"Data_(\w+)_Graph_Classified\.json", filename)
                graph_id = match.group(1) if match else filename.replace("_Graph_Classified.json", "")
                
                output_json_filename = f"Data_{graph_id}_Graph_Probs.json"
                output_json_filepath_for_graph = os.path.join(output_dir_path, output_json_filename)
                
                meta_entry = {
                    'graph_id': graph_id,
                    'original_num_edges': 0,
                    'geometry_and_labels': [],
                    'output_filepath': output_json_filepath_for_graph
                }
                current_features_tensor_for_graph = torch.empty(0, input_dim, device=DEVICE)

                try:
                    with open(json_filepath, 'r', encoding='utf-8') as f:
                        raw_graph_data_json = json.load(f)
                    
                    raw_edges_list = parse_raw_graph_data(raw_graph_data_json)
                    if not raw_edges_list: raise StopIteration("No raw edges")
                    
                    current_graph_edges_normalized = normalize_graph_edges_per_graph(raw_edges_list, args_for_current_mode.norm_method)
                    if not current_graph_edges_normalized: raise StopIteration("No normalized edges")
                    
                    if args_for_current_mode.apply_geom_noise:
                        adj_info = build_adjacency_info(current_graph_edges_normalized)
                        edges_for_features = apply_online_geometric_noise(copy.deepcopy(current_graph_edges_normalized), adj_info, args_for_current_mode)
                    else:
                        edges_for_features = current_graph_edges_normalized
                    if not edges_for_features: raise StopIteration("No edges after noise")

                    feature_dicts_list, geometry_and_labels_for_output = [], []
                    for edge_dict in edges_for_features:
                        try:
                            feats = calculate_edge_features(edge_dict['x1_norm'], edge_dict['y1_norm'], edge_dict['x2_norm'], edge_dict['y2_norm'])
                            feature_dicts_list.append(feats)
                            geometry_and_labels_for_output.append({k: edge_dict[k] for k in ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm', 'label_str']})
                        except Exception: continue
                    if not feature_dicts_list: raise StopIteration("No features calculated")

                    features_ordered_values = [[feat_d.get(col, 0.0) for col in config.FEATURE_COLS] for feat_d in feature_dicts_list]
                    features_np = np.array(features_ordered_values, dtype=np.float32)
                    if features_np.shape[0] == 0: raise StopIteration("0 features after ordering")
                    
                    scaled_features_np = scaler.transform(features_np)
                    current_features_tensor_for_graph = torch.from_numpy(scaled_features_np).to(DEVICE)
                    
                    meta_entry['original_num_edges'] = scaled_features_np.shape[0]
                    meta_entry['geometry_and_labels'] = geometry_and_labels_for_output

                except StopIteration:
                    pass
                except Exception as e_graph:
                    tqdm.write(f"  ERROR processing graph {graph_id} from {json_filepath}: {e_graph}")
                    # Print the actual traceback for one of the errors to help debug
                    if file_idx % 100 == 0: # Only print traceback occasionally to avoid spam
                        traceback.print_exc()


                current_batch_features_tensors.append(current_features_tensor_for_graph)
                current_batch_meta_info.append(meta_entry)

                is_last_file_in_list = (file_idx == len(json_files_to_process) - 1)
                if len(current_batch_features_tensors) >= batch_size or \
                   (is_last_file_in_list and current_batch_features_tensors):
                    
                    process_and_save_batch(
                        current_batch_features_tensors,
                        current_batch_meta_info,
                        pred_model,
                        DEVICE,
                        input_dim 
                    )
                    current_batch_features_tensors = [] 
                    current_batch_meta_info = []
            
    script_end_time = time.time()
    print(f"\n--- Prediction JSON Generation Finished ({script_end_time - script_start_time:.2f} seconds) ---")
    print(f"Output saved to base directory: {args.output_base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate prediction JSONs with probabilities from a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Group for script-specific arguments ---
    group_script = parser.add_argument_group('Script-Specific Arguments')
    group_script.add_argument('--model_path', type=str, 
                        default=r"cpk/best_model_orient_GraphSeq_lr1eneg04_bs128_d256_nh8_nl6_do0p1_knn16SegTh1p0_PostLN_logW_FeatNoiseOff_GeomNoiseOff_ls0p2_mlp128-64-32_OffVec.pt",
                        help='Path to the trained model (.pt file).')
    group_script.add_argument('--scaler_path', type=str,
                        default=r"dataset/bboxnorm_orient_Train_GraphSeq_scaler.pkl",
                        help='Path to the fitted scaler (.pkl file).')
    group_script.add_argument('--encoder_path', type=str,
                        default=r"dataset/bboxnorm_orient_Train_GraphSeq_label_encoder.pkl",
                        help='Path to the fitted label encoder (.pkl file).')
    group_script.add_argument('--train_input_dir', type=str,
                        default=r"C:/Users/Alireza/Desktop/New folder (4)/VF_Dataset/Train",
                        help='Path to the directory containing TRAIN *_Graph_Classified.json files.')
    group_script.add_argument('--test_input_dir', type=str,
                        default=r"C:/Users/Alireza/Desktop/New folder (4)/VF_Dataset/Test",
                        help='Path to the directory containing TEST *_Graph_Classified.json files.')
    group_script.add_argument('--output_base_dir', type=str, default="prediction_json_outputs",
                        help='Base directory to save the output JSON files.')
    group_script.add_argument('--norm_method', type=str, default="bbox_max_dim", choices=["bbox_max_dim", "normlen"],
                        help="Normalization method used for per-graph normalization.")
    group_script.add_argument('--no-noisy', action='store_true', help="If set, skip generating predictions for noisy data.")
    group_script.add_argument('--batch_size', type=int, default=512, help="Number of graphs to process in a single batch.")

    # --- Group for model architecture arguments ---
    group_model = parser.add_argument_group('Model Architecture Arguments (must match the saved model)')
    group_model.add_argument('--d_model', type=int, default=256, help='Transformer model dimension')
    group_model.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    group_model.add_argument('--num_layers', type=int, default=6, help='Number of Transformer encoder layers')
    group_model.add_argument('--dim_ff', type=int, default=1024, help='Dimension of the feedforward network')
    group_model.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    group_model.add_argument('--k_nearest', type=int, default=16, help='K for K-Nearest Lines attention')
    group_model.add_argument('--norm_first', action='store_true', help='Use Pre-LayerNormalization')
    group_model.add_argument('--mlp_head_dims', type=int, nargs='+', default=[128, 64, 32], help='Hidden layer sizes for MLP head')

    # --- Group for geometric noise arguments (MUST INCLUDE ALL from data_utils.py) ---
    group_geom_noise = parser.add_argument_group('Geometric Noise Parameters (for noisy pass)')
    group_geom_noise.add_argument('--apply_geom_noise', action='store_true', help='(Internal flag) Enable geometric noise for the noisy pass.')
    group_geom_noise.add_argument('--geom_noise_global_p', type=float, default=1.0, help='Overall probability geometric noise is applied on the noisy pass.')
    group_geom_noise.add_argument('--geom_noise_p_delete_edge', type=float, default=0.02, help='Prob of geometric edge DELETION.')
    group_geom_noise.add_argument('--geom_noise_delete_edge_ratio', type=float, default=0.02, help='Fraction of edges to DELETE.')
    group_geom_noise.add_argument('--geom_noise_p_break_edge', type=float, default=0.02, help='Prob of geometric edge BREAKING.')
    group_geom_noise.add_argument('--geom_noise_break_edge_ratio', type=float, default=0.02, help='Fraction of edges to BREAK.')
    group_geom_noise.add_argument('--geom_noise_break_length_factor', type=float, default=0.05, help='Length factor for broken edges.')
    group_geom_noise.add_argument('--geom_noise_p_angle_noise', type=float, default=0.02, help='Prob of geometric angle noise.')
    group_geom_noise.add_argument('--geom_noise_angle_std', type=float, default=0.02, help='Std deviation (radians) for angle noise.')
    group_geom_noise.add_argument('--geom_noise_p_length_noise', type=float, default=0.02, help='Prob of geometric length noise.')
    group_geom_noise.add_argument('--geom_noise_length_std', type=float, default=0.02, help='Std deviation (relative) for length noise.')
    group_geom_noise.add_argument('--geom_noise_p_delete_node', type=float, default=0.02, help='Prob of geometric node DELETION.')
    group_geom_noise.add_argument('--geom_noise_delete_node_ratio', type=float, default=0.1, help='Fraction of nodes to DELETE.')
    group_geom_noise.add_argument('--geom_noise_p_subdivide_edge', type=float, default=0.02, help='Prob of geometric edge SUBDIVISION.')
    group_geom_noise.add_argument('--geom_noise_subdivide_edge_ratio', type=float, default=0.02, help='Fraction of edges to SUBDIVIDE.')
    group_geom_noise.add_argument('--geom_noise_subdivide_n_segments', type=int, default=2, help='Number of segments for subdivision (min 2).')
    group_geom_noise.add_argument('--geom_noise_p_coord_noise', type=float, default=0.02, help='Prob of applying direct coordinate noise (Gaussian jitter).')
    group_geom_noise.add_argument('--geom_noise_coord_std', type=float, default=0.02, help='Absolute std deviation for coordinate noise.')

    args = parser.parse_args()
    
    # Set a helper attribute based on the --no-noisy flag
    args.include_noisy = not args.no_noisy

    try:
        main_generate(args)
    except Exception as e_main:
        print(f"\n--- MAIN SCRIPT FAILED ---")
        print(f"Error: {e_main}")
        traceback.print_exc()
        exit(1)
