# preprocess.py
import os
import json
import math
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pickle
import time
import sys
import torch
import traceback
import re
from collections import defaultdict

import config


def calculate_edge_features(x1n, y1n, x2n, y2n):
    """
    Calculates all edge features based on normalized, canonically ordered coordinates.
    """
    dx = x2n - x1n
    dy = y2n - y1n
    length = math.sqrt(dx**2 + dy**2)
    eps = 1e-9

    is_horizontal = 0.0
    is_vertical = 0.0
    is_positive_slope = 0.0
    is_negative_slope = 0.0
    angle_rad = 0.0
    cos_angle = 1.0
    sin_angle = 0.0

    if length > config.COORD_TOLERANCE:
        angle_rad = math.atan2(dy, dx)
        cos_angle = dx / length
        sin_angle = dy / length
        if abs(sin_angle) < config.ORIENT_TOLERANCE: is_horizontal = 1.0
        elif abs(cos_angle) < config.ORIENT_TOLERANCE: is_vertical = 1.0
        else: is_positive_slope = 1.0 if (dx * dy) > 0 else 0.0
        is_negative_slope = 1.0 if (dx * dy) < 0 else 0.0
    else:
        is_horizontal = 1.0

    abs_cos_angle = abs(cos_angle)
    abs_sin_angle = abs(sin_angle)
    mid_x_norm = (x1n + x2n) / 2.0
    mid_y_norm = (y1n + y2n) / 2.0

    cross_vec_x = sin_angle
    cross_vec_y = -cos_angle

    sum_vec_x = cos_angle + cross_vec_x
    sum_vec_y = sin_angle + cross_vec_y
    norm_sum_vec = math.sqrt(sum_vec_x**2 + sum_vec_y**2)
    norm_sum_vec_x = sum_vec_x / (norm_sum_vec + eps)
    norm_sum_vec_y = sum_vec_y / (norm_sum_vec + eps)

    s_45 = config.s_45
    norm45_x = s_45 * (cos_angle - sin_angle)
    norm45_y = s_45 * (cos_angle + sin_angle)
    norm135_x = s_45 * (sin_angle + cos_angle)
    norm135_y = s_45 * (sin_angle - cos_angle)

    features_dict = {
        'length': length, 'sin_angle': sin_angle, 'cos_angle': cos_angle,
        'x1_norm': x1n, 'y1_norm': y1n, 'x2_norm': x2n, 'y2_norm': y2n,
        'mid_x_norm': mid_x_norm, 'mid_y_norm': mid_y_norm,
        'abs_cos_angle': abs_cos_angle, 'abs_sin_angle': abs_sin_angle,
        'angle_rad': angle_rad,
        'is_horizontal': is_horizontal, 'is_vertical': is_vertical,
        'is_positive_slope': is_positive_slope, 'is_negative_slope': is_negative_slope,
        'cross_vec_x': cross_vec_x, 'cross_vec_y': cross_vec_y,
        'norm_sum_vec_x': norm_sum_vec_x, 'norm_sum_vec_y': norm_sum_vec_y,
        'norm45_x': norm45_x, 'norm45_y': norm45_y,
        'norm135_x': norm135_x, 'norm135_y': norm135_y,
    }

    points = {
        'start': (x1n, y1n),
        'mid': (mid_x_norm, mid_y_norm),
        'end': (x2n, y2n)
    }

    for point_name, (px, py) in points.items():
        for angle_name, (dx_off, dy_off) in config.OFFSET_DIRECTIONS.items():
            features_dict[f'{point_name}_{angle_name}_x'] = px + dx_off
            features_dict[f'{point_name}_{angle_name}_y'] = py + dy_off

    return {key: features_dict.get(key, 0.0) for key in config.FEATURE_COLS}


def preprocess_data(args):
    """
    Performs data preprocessing pipeline.
    """
    script_start_time = time.time()
    print(f"--- Starting Data Preprocessing ---")
    print(f"Normalization Method: {args.prep_norm_method}")
    print(f"Input directory: {args.prep_input_dir}")
    print(f"Output directory: {args.prep_output_dir}")
    print(f"JSON Pattern: {args.prep_json_pattern}")
    print(f"Output Suffix: {args.prep_output_suffix}")
    print(f"Feature Columns Used ({len(config.FEATURE_COLS)}):\n  (Features listed in config.py)")

    json_files = glob.glob(os.path.join(args.prep_input_dir, args.prep_json_pattern))
    if not json_files:
        print(f"[ERROR] No JSON files found: '{os.path.join(args.prep_input_dir, args.prep_json_pattern)}'")
        exit(1)
    print(f"Found {len(json_files)} JSON files.")

    norm_prefix = "bboxnorm" if args.prep_norm_method == "bbox_max_dim" else "normlen"
    base_filename = f"{norm_prefix}{args.prep_output_suffix}_GraphSeq"
    viz_base_filename = f"visualization_data_{norm_prefix}{args.prep_output_suffix}_GraphSeq"

    output_scaler_path = os.path.join(args.prep_output_dir, f"{base_filename}_scaler.pkl")
    output_encoder_path = os.path.join(args.prep_output_dir, f"{base_filename}_label_encoder.pkl")
    weights_output_path = os.path.join(args.prep_output_dir, f"{base_filename}_class_weights.pt")
    output_viz_json_path = os.path.join(args.prep_output_dir, f"{viz_base_filename}.json")

    all_raw_edges_for_fitting = []
    visualization_data = defaultdict(list)

    print("\n--- Processing files, ordering points, normalizing, deduplicating, and calculating features ---")
    num_files = len(json_files)
    bar_len = 40
    
    total_duplicates_removed_count = 0
    files_actually_contributing_data = 0

    for i, filepath in enumerate(json_files):
        filename = os.path.basename(filepath)
        match = re.match(r"Data_(\w+)_Graph_Classified\.json", filename)
        graph_id = match.group(1) if match else f"unknown_{i}"

        progress = (i + 1) / num_files
        filled_len = int(round(bar_len * progress))
        bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len -1) if filled_len < bar_len else '=' * bar_len
        sys.stdout.write(f"\r  Processing file {i+1}/{num_files}: [{bar}] {progress*100:.1f}% {filename[:25]}...")
        sys.stdout.flush()

        try:
            with open(filepath, 'r', encoding='utf-8') as f: graph_data_raw_json = json.load(f)
            if not isinstance(graph_data_raw_json, list):
                continue

            canonically_ordered_raw_edges_in_file = []
            nodes_for_norm_calc_in_file_set = set()
            invalid_items_in_file = 0

            for item_idx, item in enumerate(graph_data_raw_json):
                if not (isinstance(item, list) and len(item) >= 2 and
                        isinstance(item[0], list) and len(item[0]) == 2 and
                        isinstance(item[0][0], list) and len(item[0][0]) == 2 and
                        isinstance(item[0][1], list) and len(item[0][1]) == 2 and
                        isinstance(item[1], str)):
                    invalid_items_in_file += 1
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
                    elif abs(x1_canon_raw - x2_canon_raw) < config.COORD_TOLERANCE and y1_canon_raw > y2_canon_raw + config.COORD_TOLERANCE: swap_points = True
                    
                    if swap_points:
                        x1_canon_raw, y1_canon_raw, x2_canon_raw, y2_canon_raw = \
                            x2_canon_raw, y2_canon_raw, x1_canon_raw, y1_canon_raw

                    canonically_ordered_raw_edges_in_file.append({
                        'x1_raw': x1_canon_raw, 'y1_raw': y1_canon_raw,
                        'x2_raw': x2_canon_raw, 'y2_raw': y2_canon_raw,
                        'label_str': label_str
                    })
                    nodes_for_norm_calc_in_file_set.add((x1_canon_raw, y1_canon_raw))
                    nodes_for_norm_calc_in_file_set.add((x2_canon_raw, y2_canon_raw))
                except (ValueError, TypeError):
                    invalid_items_in_file += 1
                    continue
            
            if not nodes_for_norm_calc_in_file_set or not canonically_ordered_raw_edges_in_file:
                 continue

            nodes_array = np.array(list(nodes_for_norm_calc_in_file_set))
            xmin, ymin = nodes_array.min(axis=0)
            xmax, ymax = nodes_array.max(axis=0)
            width = xmax - xmin
            height = ymax - ymin
            
            if args.prep_norm_method == "bbox_max_dim":
                center_x = (xmin + xmax) / 2.0
                center_y = (ymin + ymax) / 2.0
                scale_factor = max(width / 2.0, height / 2.0, 1e-9) 
                offset_x, offset_y = center_x, center_y
            else: 
                scale_factor = 1.0
                offset_x = 0.0
                offset_y = 0.0

            seen_normalized_edges_in_graph = set() 
            num_edges_added_this_graph = 0

            for edge_raw_canon_dict in canonically_ordered_raw_edges_in_file:
                x1n_c = (edge_raw_canon_dict['x1_raw'] - offset_x) / scale_factor
                y1n_c = (edge_raw_canon_dict['y1_raw'] - offset_y) / scale_factor
                x2n_c = (edge_raw_canon_dict['x2_raw'] - offset_x) / scale_factor
                y2n_c = (edge_raw_canon_dict['y2_raw'] - offset_y) / scale_factor
                label_str = edge_raw_canon_dict['label_str']

                precision = config.NODE_COORD_PRECISION
                edge_geom_key = (
                    round(x1n_c, precision), round(y1n_c, precision),
                    round(x2n_c, precision), round(y2n_c, precision)
                )

                if edge_geom_key in seen_normalized_edges_in_graph:
                    total_duplicates_removed_count += 1
                    continue
                else:
                    seen_normalized_edges_in_graph.add(edge_geom_key)
                    clean_features_dict = calculate_edge_features(x1n_c, y1n_c, x2n_c, y2n_c)
                    all_raw_edges_for_fitting.append({**clean_features_dict, 'label_str': label_str})
                    edge_viz_info = {
                        'x1_norm': x1n_c, 'y1_norm': y1n_c,
                        'x2_norm': x2n_c, 'y2_norm': y2n_c,
                        'label_str': label_str,
                        **clean_features_dict 
                    }
                    visualization_data[graph_id].append(edge_viz_info)
                    num_edges_added_this_graph += 1
            
            if num_edges_added_this_graph > 0:
                files_actually_contributing_data += 1

        except json.JSONDecodeError:
            print(f"\n[ERROR] Decoding JSON failed for {filename}. Skipping.")
            continue
        except Exception as e:
            print(f"\n[ERROR] Processing file {filename}: {e}")
            traceback.print_exc()
            continue
    
    sys.stdout.write('\r' + ' ' * (bar_len + 70) + '\r') 
    print(f"\n--- File Reading & Deduplication Complete ---")
    print(f"  {num_files} files scanned.")
    print(f"  {files_actually_contributing_data} files contributed unique edge data to the output.")
    print(f"  Total duplicate line geometries removed: {total_duplicates_removed_count}")

    if not all_raw_edges_for_fitting:
        print("[ERROR] No edge data collected from any file after deduplication. Exiting.")
        exit(1)

    print(f"\nFitting LabelEncoder on {len(all_raw_edges_for_fitting)} unique edge instances...")
    all_labels_str = [edge['label_str'] for edge in all_raw_edges_for_fitting]
    label_encoder = LabelEncoder()
    try:
        label_encoder.fit(all_labels_str)
        num_classes = len(label_encoder.classes_)
        print(f"Found {num_classes} unique labels: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"[ERROR] Failed to fit LabelEncoder: {e}")
        exit(1)

    print(f"Fitting StandardScaler on {len(config.FEATURE_COLS)} features from {len(all_raw_edges_for_fitting)} unique edge instances...")
    features_for_fitting_list = [[edge_dict.get(col, 0.0) for col in config.FEATURE_COLS] for edge_dict in all_raw_edges_for_fitting]
    try:
        features_np_for_fitting = np.array(features_for_fitting_list, dtype=np.float32)
        if features_np_for_fitting.ndim != 2 or features_np_for_fitting.shape[1] != len(config.FEATURE_COLS):
            print(f"[ERROR] Incorrect shape for scaler fitting: {features_np_for_fitting.shape}. Expected (N, {len(config.FEATURE_COLS)}).")
            exit(1)
        scaler = StandardScaler()
        scaler.fit(features_np_for_fitting)
        print(f"StandardScaler fitted.")
    except Exception as e:
        print(f"[ERROR] Failed to fit StandardScaler: {e}")
        traceback.print_exc()
        exit(1)

    print("Calculating class weights (from unique, clean data)...")
    try:
        encoded_labels_for_weights = label_encoder.transform(all_labels_str)
        classes_to_compute = np.arange(num_classes)
        raw_weights = compute_class_weight('balanced', classes=classes_to_compute, y=encoded_labels_for_weights)
        processed_weights = np.log1p(raw_weights)
        class_weights_tensor = torch.tensor(processed_weights, dtype=torch.float)
        print(f"Calculated Logarithmic (log1p) class weights: {class_weights_tensor.numpy().round(4)}")
        print(f"Saving class weights to {weights_output_path}...")
        torch.save(class_weights_tensor, weights_output_path)
    except ValueError as e:
        print(f"[ERROR] ValueError during weight calculation ({e}). Saving uniform weights.")
        class_weights_tensor = torch.full((num_classes,), math.log1p(1.0), dtype=torch.float)
        torch.save(class_weights_tensor, weights_output_path)
    except Exception as e:
        print(f"[ERROR] Could not calculate/save class weights: {e}. Saving uniform weights.")
        class_weights_tensor = torch.full((num_classes,), math.log1p(1.0), dtype=torch.float)
        torch.save(class_weights_tensor, weights_output_path)

    config.ensure_dirs(args)

    print(f"\nSaving scaler to {output_scaler_path}...")
    try:
        with open(output_scaler_path, 'wb') as f: pickle.dump(scaler, f)
        print("Scaler saved.")
    except Exception as e: print(f"[ERROR] Could not save Scaler: {e}")

    print(f"Saving label encoder to {output_encoder_path}...")
    try:
        with open(output_encoder_path, 'wb') as f: pickle.dump(label_encoder, f)
        print("Label encoder saved.")
    except Exception as e: print(f"[ERROR] Could not save Label Encoder: {e}")

    print(f"Saving combined data/visualization JSON to {output_viz_json_path}...")
    try:
        viz_data_dict = dict(visualization_data)
        with open(output_viz_json_path, 'w', encoding='utf-8') as f:
            json.dump(viz_data_dict, f) 
        print("Data/Visualization JSON saved.")
    except Exception as e:
        print(f"[ERROR] Could not save visualization JSON: {e}")

    script_end_time = time.time()
    print(f"\n--- Preprocessing Script Finished ({script_end_time - script_start_time:.2f} seconds) ---")

if __name__ == "__main__":
    try:
        args = config.setup_arg_parser()
        preprocess_data(args)
    except Exception as main_e:
        print(f"\n[FATAL ERROR] An unhandled exception occurred in main execution: {main_e}")
        traceback.print_exc()
        exit(1)
