# refiner_preprocess.py
import os
import json
import math
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
import time
import sys
import torch
import traceback
import re
from collections import defaultdict
from tqdm import tqdm

import refiner_config as r_config
from refiner_config import calculate_geometric_features_for_refiner


def preprocess_refiner_data_split(args, input_dir, output_suffix, num_classes_orig_model,
                                  existing_label_encoder=None, existing_scaler=None, is_training_split=True):
    print(f"\nStarting Refiner Preprocessing for Split (Output Suffix: {output_suffix})")
    print(f"  Input directory (original model outputs): {input_dir}")
    print(f"  JSON Pattern: {args.prep_json_pattern_orig_model_output}")

    json_files = glob.glob(os.path.join(input_dir, args.prep_json_pattern_orig_model_output))
    if not json_files:
        print(f"  [WARN] No JSON files found in '{input_dir}' with pattern '{args.prep_json_pattern_orig_model_output}'. Skipping this split.")
        return None, None, None

    print(f"  Found {len(json_files)} JSON files for this split.")

    manifest_base_filename = f"refiner_geom{output_suffix}_RefinerData_manifest"
    output_manifest_json_path = os.path.join(args.prep_output_dir, f"{manifest_base_filename}.json")
    individual_graph_data_dir = os.path.join(args.prep_output_dir, "graph_data")
    os.makedirs(individual_graph_data_dir, exist_ok=True)

    all_geom_features_for_scaler_fitting = []
    label_encoder = existing_label_encoder
    scaler = existing_scaler

    if is_training_split and label_encoder is None:
        print("  Fitting new LabelEncoder for refiner (on ground truth labels)...")
        temp_all_gt_labels = []
        for filepath_le in tqdm(json_files, desc="Reading files for label fitting", ncols=100):
            try:
                with open(filepath_le, 'r', encoding='utf-8') as f: graph_output_data = json.load(f)
                if not isinstance(graph_output_data, list): continue
                for edge_data in graph_output_data:
                    if isinstance(edge_data, dict) and 'label_str' in edge_data:
                        temp_all_gt_labels.append(edge_data['label_str'])
            except Exception as e_le:
                print(f"    Warning: Could not read {os.path.basename(filepath_le)} for label fitting: {e_le}")
        
        if not temp_all_gt_labels: 
            print("[ERROR] No ground truth labels found. Exiting.")
            sys.exit(1)

        label_encoder = LabelEncoder()
        label_encoder.fit(temp_all_gt_labels)
        print(f"  Refiner LabelEncoder fitted. Classes: {list(label_encoder.classes_)}")
        if len(label_encoder.classes_) != num_classes_orig_model:
            print(f"  [WARNING] Encoder classes ({len(label_encoder.classes_)}) != --num_classes_orig_model ({num_classes_orig_model}).")
    
    elif label_encoder is None: 
        print("[ERROR] LabelEncoder must be provided. Exiting.")
        sys.exit(1)

    r_config.update_feature_columns(len(label_encoder.classes_))
    print(f"  Updated r_config feature columns for {len(label_encoder.classes_)} classes.")

    print(f"  Processing files for split '{output_suffix}'...")
    num_edges_processed_total = 0
    processed_graph_ids_for_manifest = []

    pbar = tqdm(json_files, desc=f"Processing '{output_suffix}' files", ncols=120)
    for filepath in pbar:
        filename = os.path.basename(filepath)
        match = re.match(r"Data_([a-zA-Z0-9_]+)_Graph_Probs\.json", filename)
        graph_id = match.group(1) if match else f"unknown_graph_{filepath}"

        try:
            with open(filepath, 'r', encoding='utf-8') as f: graph_output_data_from_orig_model = json.load(f)
            if not isinstance(graph_output_data_from_orig_model, list) or not graph_output_data_from_orig_model: continue

            edges_for_current_graph_json = []
            for edge_data_orig in graph_output_data_from_orig_model:
                if not isinstance(edge_data_orig, dict) or not all(k in edge_data_orig for k in ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm', 'label_str', 'probabilities']): continue

                x1n, y1n, x2n, y2n = edge_data_orig['x1_norm'], edge_data_orig['y1_norm'], edge_data_orig['x2_norm'], edge_data_orig['y2_norm']
                gt_label_str, orig_model_probs = edge_data_orig['label_str'], edge_data_orig['probabilities']
                if len(orig_model_probs) != len(label_encoder.classes_): continue

                geom_features_unscaled = calculate_geometric_features_for_refiner(x1n, y1n, x2n, y2n)
                prob_features = {r_config.PROB_FEATURE_COLS[k]: orig_model_probs[k] for k in range(len(orig_model_probs))}
                orig_model_pred_idx = np.argmax(orig_model_probs)
                onehot_features = {r_config.ONEHOT_FEATURE_COLS[k]: (1.0 if k == orig_model_pred_idx else 0.0) for k in range(len(label_encoder.classes_))}

                try: gt_label_idx = label_encoder.transform([gt_label_str])[0]
                except ValueError: continue

                is_orig_correct = 1 if gt_label_idx == orig_model_pred_idx else 0
                edge_entry = {'x1_norm': x1n, 'y1_norm': y1n, 'x2_norm': x2n, 'y2_norm': y2n, **geom_features_unscaled, **prob_features, **onehot_features,
                              r_config.LABEL_COL_STR_REF: gt_label_str, r_config.TARGET_COL_REF: int(gt_label_idx), r_config.ORIG_PRED_IDX_COL_REF: int(orig_model_pred_idx),
                              r_config.CORRECTNESS_TARGET_COL_REF: int(is_orig_correct), 'original_probabilities_list': orig_model_probs}
                edges_for_current_graph_json.append(edge_entry)
                if is_training_split and scaler is None: all_geom_features_for_scaler_fitting.append(geom_features_unscaled)

            if edges_for_current_graph_json:
                individual_graph_json_path = os.path.join(individual_graph_data_dir, f"{graph_id}.json")
                with open(individual_graph_json_path, 'w', encoding='utf-8') as f_graph: json.dump(edges_for_current_graph_json, f_graph)
                processed_graph_ids_for_manifest.append(graph_id)
                num_edges_processed_total += len(edges_for_current_graph_json)
                pbar.set_postfix(graphs=f"{len(processed_graph_ids_for_manifest)}", edges=f"{num_edges_processed_total}")
        
        except Exception as e: 
            print(f"\n[ERROR] processing file {filename}: {e}")
            traceback.print_exc()

    print(f"\n  Finished reading files for split '{output_suffix}'. Processed {num_edges_processed_total} edges across {len(processed_graph_ids_for_manifest)} graphs.")
    if not processed_graph_ids_for_manifest: return None, scaler, label_encoder

    if is_training_split and scaler is None:
        if not all_geom_features_for_scaler_fitting: 
            print("  [ERROR] No features to fit StandardScaler.")
            sys.exit(1)

        print(f"  Fitting new StandardScaler on {len(all_geom_features_for_scaler_fitting)} instances...")
        features_np = np.array([[g.get(col, 0.0) for col in r_config.GEOM_FEATURE_COLS] for g in all_geom_features_for_scaler_fitting], dtype=np.float32)
        
        if features_np.ndim != 2 or features_np.shape[1] != len(r_config.GEOM_FEATURE_COLS): 
            print(f"  [ERROR] Feature array shape error.")
            sys.exit(1)
        
        scaler = StandardScaler()
        scaler.fit(features_np)
        if hasattr(scaler, 'scale_') and np.any(scaler.scale_ < 1e-7):
            problematic_indices = np.where(scaler.scale_ < 1e-7)[0]
            print(f"  [WARNING] Scaler has zero/small std dev for features at indices: {problematic_indices}. Setting their scale to 1.0.")
            scaler.scale_[problematic_indices] = 1.0
        print(f"  StandardScaler fitted.")

    print(f"  Saving refiner manifest for split '{output_suffix}' to {output_manifest_json_path}...")
    try:
        with open(output_manifest_json_path, 'w', encoding='utf-8') as f_manifest:
            json.dump(processed_graph_ids_for_manifest, f_manifest, indent=2)
        print(f"  Successfully saved manifest with {len(processed_graph_ids_for_manifest)} graph IDs.")
    except Exception as e: 
        print(f"  [ERROR] Could not save manifest: {e}")
        return None, scaler, label_encoder

    return output_manifest_json_path, scaler, label_encoder


def main_preprocess_refiner(args_obj):
    script_start_time = time.time()
    print("Starting Refiner Data Preprocessing Overall")
    r_config.ensure_dirs(args_obj)
    train_manifest_path, scaler, label_encoder = preprocess_refiner_data_split(
        args_obj, input_dir=args_obj.prep_input_dir_orig_model_output, output_suffix=args_obj.prep_output_suffix_train,
        num_classes_orig_model=args_obj.num_classes_orig_model, is_training_split=True
    )
    if not train_manifest_path or not scaler or not label_encoder: 
        print("[FATAL] Preprocessing for training data failed.")
        sys.exit(1)

    output_scaler_path = os.path.join(args_obj.prep_output_dir, f"{args_obj.scaler_base_name}_scaler.pkl")
    output_encoder_path = os.path.join(args_obj.prep_output_dir, f"{args_obj.encoder_base_name}_label_encoder.pkl")
    print(f"\nSaving globally fitted scaler to {output_scaler_path}...")
    with open(output_scaler_path, 'wb') as f: pickle.dump(scaler, f)
    print(f"Saving globally fitted label encoder to {output_encoder_path}...")
    with open(output_encoder_path, 'wb') as f: pickle.dump(label_encoder, f)

    test_manifest_path, _, _ = preprocess_refiner_data_split(
        args_obj, input_dir=args_obj.prep_input_dir_orig_model_output_test, output_suffix=args_obj.prep_output_suffix_test,
        num_classes_orig_model=args_obj.num_classes_orig_model, existing_label_encoder=label_encoder, existing_scaler=scaler, is_training_split=False
    )
    if not test_manifest_path: print("[WARNING] Preprocessing for test data failed.")

    print(f"\nRefiner Preprocessing Script Finished ({time.time() - script_start_time:.2f} seconds)")
    print(f"  Train Manifest: {train_manifest_path or 'Not generated'}")
    print(f"  Test Manifest: {test_manifest_path or 'Not generated'}")
    print(f"  Scaler/Encoder saved to: {args_obj.prep_output_dir}")

if __name__ == "__main__":
    try:
        parsed_args = r_config.setup_arg_parser()
        r_config.args_parsed = parsed_args
        main_preprocess_refiner(parsed_args)
    except Exception as main_e:
        print(f"\n[FATAL ERROR] {main_e}")
        traceback.print_exc()
        sys.exit(1)
