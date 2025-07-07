# refiner_data_utils.py
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import pickle
import os
import json
import random
import math
from tqdm import tqdm
import functools
import argparse
import traceback
import copy
from collections import defaultdict

import refiner_config as r_config
from refiner_config import calculate_geometric_features_for_refiner


PAD_VALUE_FEATURES_REF = 0.0
PAD_VALUE_LABELS_REF = -100
PAD_VALUE_ORIG_PRED_IDX_REF = -100
PAD_VALUE_CORRECTNESS_REF = -100


def build_adjacency_info(graph_edges):
    node_to_id = {}
    id_to_node = {}
    edge_idx_to_node_pair = []
    node_to_edges = defaultdict(list)
    precision = r_config.NODE_COORD_PRECISION_REF
    
    def get_node_id(x, y):
        coord_tuple = (round(x, precision), round(y, precision))
        if coord_tuple not in node_to_id:
            new_id = len(node_to_id)
            node_to_id[coord_tuple] = new_id
            id_to_node[new_id] = coord_tuple
        return node_to_id[coord_tuple]
    
    for i, edge in enumerate(graph_edges):
        try:
            node1_id = get_node_id(edge['x1_norm'], edge['y1_norm'])
            node2_id = get_node_id(edge['x2_norm'], edge['y2_norm'])
            edge_idx_to_node_pair.append((node1_id, node2_id))
            node_to_edges[node1_id].append(i)
            node_to_edges[node2_id].append(i)
        except KeyError:
            edge_idx_to_node_pair.append((-1, -1))
    return {'node_to_id': node_to_id, 'id_to_node': id_to_node,
            'edge_idx_to_node_pair': edge_idx_to_node_pair, 'node_to_edges': node_to_edges}


def apply_online_geometric_noise_refiner(graph_edges_input_dicts, adj_info, args):
    if not args.apply_geom_noise or random.random() >= args.geom_noise_global_p:
        return graph_edges_input_dicts
    current_edges = copy.deepcopy(graph_edges_input_dicts)
    if random.random() < args.geom_noise_p_coord_noise and args.geom_noise_coord_std > 0:
        coord_std = args.geom_noise_coord_std
        for edge_dict in current_edges:
            if 'x1_norm' in edge_dict: edge_dict['x1_norm'] += random.gauss(0, coord_std)
            if 'y1_norm' in edge_dict: edge_dict['y1_norm'] += random.gauss(0, coord_std)
            if 'x2_norm' in edge_dict: edge_dict['x2_norm'] += random.gauss(0, coord_std)
            if 'y2_norm' in edge_dict: edge_dict['y2_norm'] += random.gauss(0, coord_std)
    return current_edges


class RefinerGraphSequenceDataset(Dataset):
    def __init__(self, manifest_file_path, prep_output_dir):
        self.manifest_file_path = manifest_file_path
        self.prep_output_dir = prep_output_dir
        self.graph_ids_list = []
        try:
            with open(manifest_file_path, 'r', encoding='utf-8') as f:
                self.graph_ids_list = json.load(f)
            if not self.graph_ids_list:
                raise ValueError(f"No graph IDs found in manifest file: '{manifest_file_path}'.")
            print(f"  RefinerDataset: Loaded {len(self.graph_ids_list)} graph IDs from {os.path.basename(manifest_file_path)}.")
        except FileNotFoundError: raise FileNotFoundError(f"Refiner manifest file not found: '{manifest_file_path}'")
        except json.JSONDecodeError: raise IOError(f"Error decoding JSON from refiner manifest file: '{manifest_file_path}'")
        except Exception as e: raise IOError(f"Error loading refiner graph IDs from manifest '{manifest_file_path}': {e}")
    
    def __len__(self): return len(self.graph_ids_list)
    
    def __getitem__(self, idx):
        if idx >= len(self.graph_ids_list): raise IndexError("Index out of bounds.")
        graph_id = self.graph_ids_list[idx]
        individual_graph_json_path = os.path.join(self.prep_output_dir, "graph_data", f"{graph_id}.json")
        try:
            with open(individual_graph_json_path, 'r', encoding='utf-8') as f_graph:
                return json.load(f_graph)
        except Exception: return []


def refiner_collate_fn(batch, args, scaler, num_total_classes_from_encoder, is_eval=False):
    feature_sequences, gt_label_sequences, orig_pred_idx_sequences, correctness_target_sequences = [], [], [], []
    selected_feature_cols = list(r_config.GEOM_FEATURE_COLS)
    if args.refiner_input_feature_mode == "probs_and_onehot":
        selected_feature_cols.extend(r_config.PROB_FEATURE_COLS)
        selected_feature_cols.extend(r_config.ONEHOT_FEATURE_COLS)
    elif args.refiner_input_feature_mode == "probs_only": selected_feature_cols.extend(r_config.PROB_FEATURE_COLS)
    elif args.refiner_input_feature_mode == "onehot_only": selected_feature_cols.extend(r_config.ONEHOT_FEATURE_COLS)

    if len(r_config.PROB_FEATURE_COLS) != num_total_classes_from_encoder:
        r_config.update_feature_columns(num_total_classes_from_encoder)
        selected_feature_cols = list(r_config.GEOM_FEATURE_COLS)
        if args.refiner_input_feature_mode == "probs_and_onehot":
            selected_feature_cols.extend(r_config.PROB_FEATURE_COLS)
            selected_feature_cols.extend(r_config.ONEHOT_FEATURE_COLS)
        elif args.refiner_input_feature_mode == "probs_only": selected_feature_cols.extend(r_config.PROB_FEATURE_COLS)
        elif args.refiner_input_feature_mode == "onehot_only": selected_feature_cols.extend(r_config.ONEHOT_FEATURE_COLS)

    for graph_edge_dicts_from_json in batch:
        if not graph_edge_dicts_from_json:
            feature_sequences.append(torch.empty((0, len(selected_feature_cols)), dtype=torch.float))
            gt_label_sequences.append(torch.empty((0,), dtype=torch.long))
            orig_pred_idx_sequences.append(torch.empty((0,), dtype=torch.long))
            correctness_target_sequences.append(torch.empty((0,), dtype=torch.float))
            continue
        
        adj_info_for_geom_noise = build_adjacency_info(graph_edge_dicts_from_json)
        graph_edge_dicts_after_geom_noise = apply_online_geometric_noise_refiner(graph_edge_dicts_from_json, adj_info_for_geom_noise, args)
        
        temp_edges_recalculated_geom = []
        for edge_dict_noisy_coords in graph_edge_dicts_after_geom_noise:
            processed_edge_dict = edge_dict_noisy_coords.copy()
            new_geom_features = calculate_geometric_features_for_refiner(processed_edge_dict['x1_norm'], processed_edge_dict['y1_norm'], processed_edge_dict['x2_norm'], processed_edge_dict['y2_norm'])
            processed_edge_dict.update(new_geom_features)
            temp_edges_recalculated_geom.append(processed_edge_dict)
        graph_edge_dicts_for_corruption = temp_edges_recalculated_geom

        if not is_eval and args.label_corruption_frac_graphs > 0 and args.label_corruption_frac_edges > 0 and random.random() < args.label_corruption_frac_graphs:
            num_edges = len(graph_edge_dicts_for_corruption)
            if num_edges > 0:
                indices_to_corrupt = [i for i in range(num_edges) if random.random() < args.label_corruption_frac_edges]
                for idx_corr in indices_to_corrupt:
                    edge_to_corrupt = graph_edge_dicts_for_corruption[idx_corr]
                    gt_idx = edge_to_corrupt[r_config.TARGET_COL_REF]
                    possible_new_indices = [l_idx for l_idx in range(num_total_classes_from_encoder) if l_idx != gt_idx]
                    if not possible_new_indices: possible_new_indices = list(range(num_total_classes_from_encoder))
                    corrupted_pred_idx = random.choice(possible_new_indices) if possible_new_indices else edge_to_corrupt[r_config.ORIG_PRED_IDX_COL_REF]
                    edge_to_corrupt[r_config.ORIG_PRED_IDX_COL_REF] = corrupted_pred_idx
                    if any(s in args.refiner_input_feature_mode for s in ["probs", "onehot"]):
                        for k in range(num_total_classes_from_encoder):
                            edge_to_corrupt[r_config.ONEHOT_FEATURE_COLS_TEMPLATE.format(k)] = 1.0 if k == corrupted_pred_idx else 0.0
                            if num_total_classes_from_encoder > 1:
                                prob_main, prob_others = 0.9, (1.0 - 0.9) / (num_total_classes_from_encoder - 1)
                                edge_to_corrupt[r_config.PROB_FEATURE_COLS_TEMPLATE.format(k)] = prob_main if k == corrupted_pred_idx else prob_others
                            else: edge_to_corrupt[r_config.PROB_FEATURE_COLS_TEMPLATE.format(k)] = 1.0
                    edge_to_corrupt[r_config.CORRECTNESS_TARGET_COL_REF] = 1 if gt_idx == corrupted_pred_idx else 0

        graph_features, graph_gt_labels, graph_orig_preds, graph_corr_targets = [], [], [], []
        for edge_dict in graph_edge_dicts_for_corruption:
            geom_unscaled = [edge_dict.get(g, PAD_VALUE_FEATURES_REF) for g in r_config.GEOM_FEATURE_COLS]
            scaled_geom = scaler.transform(np.array(geom_unscaled, dtype=np.float32).reshape(1, -1)).flatten()
            final_features = edge_dict.copy()
            for i, g_col in enumerate(r_config.GEOM_FEATURE_COLS): final_features[g_col] = scaled_geom[i]
            graph_features.append([final_features.get(f, PAD_VALUE_FEATURES_REF) for f in selected_feature_cols])
            graph_gt_labels.append(edge_dict[r_config.TARGET_COL_REF])
            graph_orig_preds.append(edge_dict[r_config.ORIG_PRED_IDX_COL_REF])
            graph_corr_targets.append(float(edge_dict[r_config.CORRECTNESS_TARGET_COL_REF]))

        feature_sequences.append(torch.tensor(graph_features, dtype=torch.float))
        gt_label_sequences.append(torch.tensor(graph_gt_labels, dtype=torch.long))
        orig_pred_idx_sequences.append(torch.tensor(graph_orig_preds, dtype=torch.long))
        correctness_target_sequences.append(torch.tensor(graph_corr_targets, dtype=torch.float))

    features_padded = rnn_utils.pad_sequence(feature_sequences, batch_first=True, padding_value=PAD_VALUE_FEATURES_REF)
    gt_labels_padded = rnn_utils.pad_sequence(gt_label_sequences, batch_first=True, padding_value=PAD_VALUE_LABELS_REF)
    orig_pred_idx_padded = rnn_utils.pad_sequence(orig_pred_idx_sequences, batch_first=True, padding_value=PAD_VALUE_ORIG_PRED_IDX_REF)
    correctness_targets_padded = rnn_utils.pad_sequence(correctness_target_sequences, batch_first=True, padding_value=PAD_VALUE_CORRECTNESS_REF)
    lengths = torch.tensor([s.shape[0] for s in feature_sequences], dtype=torch.long)
    attention_mask = torch.zeros(features_padded.shape[:2], dtype=torch.bool, device=features_padded.device)
    for i, l in enumerate(lengths): attention_mask[i, l:] = True
    return features_padded, gt_labels_padded, orig_pred_idx_padded, correctness_targets_padded, attention_mask


def load_refiner_scaler_encoder(args):
    try:
        with open(args.scaler_path, 'rb') as f: scaler = pickle.load(f)
        with open(args.encoder_path, 'rb') as f: label_encoder = pickle.load(f)
        num_classes = len(label_encoder.classes_)
        print(f"Loaded Refiner Scaler (geom-only) from {args.scaler_path}")
        print(f"Loaded Refiner LabelEncoder from {args.encoder_path}. Classes: {list(label_encoder.classes_)} ({num_classes} classes)")
        r_config.update_feature_columns(num_classes)
        return scaler, label_encoder, num_classes
    except Exception as e:
        print(f"FATAL: Error loading refiner scaler/encoder: {e}. Ensure refiner_preprocess.py has been run.")
        raise


def load_refiner_data(args, scaler, label_encoder, is_eval=False):
    manifest_file_path = args.eval_data_json_path if is_eval else args.train_val_data_json_path
    num_classes = len(label_encoder.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Refiner graph ID manifest from {manifest_file_path}...")
    try:
        full_dataset = RefinerGraphSequenceDataset(manifest_file_path=manifest_file_path, prep_output_dir=args.dataset_dir)
        collate_fn_args = {'args': argparse.Namespace(**vars(args)), 'scaler': scaler, 'num_total_classes_from_encoder': num_classes}

        if is_eval:
            collate_fn_args['args'].apply_geom_noise = args.eval_with_geom_noise
            collate_fn_args['args'].label_corruption_frac_graphs = 0.0
            collate_fn_args['is_eval'] = True
            current_collate_fn = functools.partial(refiner_collate_fn, **collate_fn_args)
            loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=current_collate_fn, persistent_workers=args.num_workers > 0)
            return loader, None, None, None

        total_size = len(full_dataset)
        if total_size == 0: raise ValueError("No data in manifest for train/val.")
        val_size = int(total_size * args.val_split)
        train_size = total_size - val_size
        if train_size <= 0 or (args.val_split > 0 and val_size <= 0): raise ValueError("Invalid split sizes.")
        
        train_dataset, val_dataset_obj = random_split(full_dataset, [train_size, val_size]) if val_size > 0 else (full_dataset, None)
        print(f"Refiner data: Total={total_size}, Train={len(train_dataset)}, Val={len(val_dataset_obj) if val_dataset_obj else 0}")

        print("Calculating class weights for Refiner's main head (from training split GT)...")
        all_train_gt_labels = []
        for i in tqdm(range(len(train_dataset)), desc="Extracting GT labels", leave=False, ncols=100):
            graph_edges = train_dataset[i]
            if graph_edges: all_train_gt_labels.extend([e[r_config.TARGET_COL_REF] for e in graph_edges])
        
        class_weights_tensor = None
        if not all_train_gt_labels:
            class_weights_tensor = torch.ones(num_classes, dtype=torch.float)
        else:
            from sklearn.utils.class_weight import compute_class_weight
            try:
                raw_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=np.array(all_train_gt_labels))
                processed_weights = np.log1p(raw_weights)
                if args.max_class_weight is not None: processed_weights = np.clip(processed_weights, a_min=None, a_max=args.max_class_weight)
                class_weights_tensor = torch.tensor(processed_weights, dtype=torch.float)
            except ValueError: class_weights_tensor = torch.ones(num_classes, dtype=torch.float)
        
        # Correctness head weight calculation
        print("Calculating pos_weight for Refiner's correctness head (from training split)...")
        all_train_corr_targets = []
        for i in tqdm(range(len(train_dataset)), desc="Extracting Corr. targets", leave=False, ncols=100):
            graph_edges = train_dataset[i]
            if graph_edges: all_train_corr_targets.extend([e[r_config.CORRECTNESS_TARGET_COL_REF] for e in graph_edges])
        
        correctness_pos_weight = None
        if args.correctness_loss_weighting_mode == 'none':
            print("  Correctness head loss weighting is DISABLED.")
        elif not all_train_corr_targets:
            print("  Warning: No correctness targets found. Cannot compute pos_weight.")
        else:
            targets_np = np.array(all_train_corr_targets)
            num_incorrect = (targets_np == 0).sum()
            num_correct = (targets_np == 1).sum()

            if num_correct > 0 and num_incorrect > 0:
                pos_weight_val = 0.0
                if args.correctness_loss_weighting_mode == 'linear':
                    pos_weight_val = num_incorrect / num_correct
                    print(f"  Mode 'linear': {num_incorrect} Incorrect / {num_correct} Correct. Raw pos_weight = {pos_weight_val:.4f}")
                elif args.correctness_loss_weighting_mode == 'log':
                    pos_weight_val = math.log1p(num_incorrect / num_correct)
                    print(f"  Mode 'log': log1p({num_incorrect} / {num_correct}). Raw pos_weight = {pos_weight_val:.4f}")
                elif args.correctness_loss_weighting_mode == 'manual':
                    pos_weight_val = args.correctness_loss_manual_pos_weight
                    print(f"  Mode 'manual': Using specified pos_weight = {pos_weight_val:.4f}")

                if args.correctness_loss_weighting_mode in ['linear', 'log'] and args.max_correctness_pos_weight is not None:
                    if pos_weight_val > args.max_correctness_pos_weight:
                        print(f"  Clipping pos_weight from {pos_weight_val:.4f} to max value {args.max_correctness_pos_weight}")
                        pos_weight_val = args.max_correctness_pos_weight
                
                correctness_pos_weight = torch.tensor([pos_weight_val], dtype=torch.float)
            else:
                print(f"  Warning: Only one class present for correctness head. Skipping pos_weight calculation.")
        
        if correctness_pos_weight is not None:
             print(f"Refiner correctness head final pos_weight: {correctness_pos_weight.item():.4f}")

        train_collate = functools.partial(refiner_collate_fn, **{**collate_fn_args, 'is_eval': False})
        val_collate_args = collate_fn_args.copy()
        val_collate_args['args'].apply_geom_noise = False
        val_collate_args['args'].label_corruption_frac_graphs = 0.0
        val_collate = functools.partial(refiner_collate_fn, **{**val_collate_args, 'is_eval': True})
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=train_collate, persistent_workers=args.num_workers > 0)
        val_loader = DataLoader(val_dataset_obj, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=val_collate, persistent_workers=args.num_workers > 0) if val_dataset_obj else None
        
        return train_loader, val_loader, class_weights_tensor.to(device), correctness_pos_weight

    except Exception as e:
        print(f"FATAL Error loading/splitting refiner data from {manifest_file_path}: {e}")
        traceback.print_exc()
        return None, None, None, None
