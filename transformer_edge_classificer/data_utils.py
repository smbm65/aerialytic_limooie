# data_utils.py
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
import numpy as np
import pickle
import os
import json
import random
import math
from tqdm import tqdm
import functools
from collections import defaultdict
import argparse
import traceback
import copy

import config
from preprocess import calculate_edge_features

# Constants
PAD_VALUE_FEATURES = 0.0
PAD_VALUE_LABELS = -100

# Dataset Class
class GraphSequenceDataset(Dataset):
    def __init__(self, data_path, graph_ids=None):
        self.data_path = data_path
        self.graph_data = {}
        self.graph_id_list = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f: all_data = json.load(f)
            if graph_ids is None: self.graph_id_list = sorted(all_data.keys())
            else: self.graph_id_list = [str(gid) for gid in graph_ids if str(gid) in all_data]
            self.graph_data = {gid: all_data[gid] for gid in self.graph_id_list}
            if not self.graph_data: raise ValueError(f"No graphs found for the provided IDs in '{data_path}'.")
        except FileNotFoundError: raise FileNotFoundError(f"Data file not found: '{data_path}'")
        except json.JSONDecodeError: raise IOError(f"Error decoding JSON from file: '{data_path}'")
        except Exception as e: raise IOError(f"An unexpected error occurred while loading graph data from '{data_path}': {e}")
    def __len__(self): return len(self.graph_id_list)
    def __getitem__(self, idx):
        if idx >= len(self.graph_id_list): raise IndexError("Index out of bounds in GraphSequenceDataset.")
        graph_id = self.graph_id_list[idx]
        return list(self.graph_data[graph_id]) # Return a copy as a list

# Adjacency Builder
def build_adjacency_info(graph_edges):
    node_to_id = {}
    id_to_node = {}
    edge_idx_to_node_pair = []
    node_to_edges = defaultdict(list)
    precision = config.NODE_COORD_PRECISION
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
            # This can happen if an edge dict is malformed (e.g., during noise processing)
            edge_idx_to_node_pair.append((-1, -1)) # Placeholder for invalid edge
    return {'node_to_id': node_to_id, 'id_to_node': id_to_node,
            'edge_idx_to_node_pair': edge_idx_to_node_pair, 'node_to_edges': node_to_edges}


# Geometric Noise Function (Already efficient for its purpose)
def apply_online_geometric_noise(graph_edges_input, adj_info, args):
    """
    Applies geometric noise based on probabilities and ratios in args.
    Order: Subdiv -> NodeDel -> EdgeDel -> Break -> Angle/Len -> Coord.
    Returns a new list of edge dictionaries. This function is already designed
    to be as efficient as possible for complex, non-vectorizable graph operations.
    """
    if not args.apply_geom_noise:
        return graph_edges_input
    if random.random() >= args.geom_noise_global_p: # Overall probability check
        return graph_edges_input

    num_edges_initial = len(graph_edges_input)
    if num_edges_initial == 0:
        return graph_edges_input

    # Operate on a deep copy to avoid modifying the original list passed to the function
    current_edges = copy.deepcopy(graph_edges_input)
    num_nodes = len(adj_info['id_to_node']) # Initial number of nodes

    output_edges_after_subdivision = []
    subdivided_original_indices = set() # Store original indices of edges that were subdivided

    # 1. Edge Subdivision
    if random.random() < args.geom_noise_p_subdivide_edge and args.geom_noise_subdivide_edge_ratio > 0:
        n_segments = max(2, args.geom_noise_subdivide_n_segments)
        num_to_subdivide_target = int(num_edges_initial * args.geom_noise_subdivide_edge_ratio)
        
        actual_num_to_subdivide = min(num_to_subdivide_target, num_edges_initial)
        if actual_num_to_subdivide > 0:
            indices_to_subdivide = random.sample(range(num_edges_initial), k=actual_num_to_subdivide)

            for idx_orig in indices_to_subdivide:
                edge = current_edges[idx_orig]
                if not all(k in edge for k in ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm', 'label_str']):
                    continue

                x1, y1, x2, y2 = edge['x1_norm'], edge['y1_norm'], edge['x2_norm'], edge['y2_norm']
                label_str = edge['label_str']

                xs = np.linspace(x1, x2, n_segments + 1)
                ys = np.linspace(y1, y2, n_segments + 1)

                for k_seg in range(n_segments):
                    new_sub_edge = {
                        'x1_norm': xs[k_seg], 'y1_norm': ys[k_seg],
                        'x2_norm': xs[k_seg+1], 'y2_norm': ys[k_seg+1],
                        'label_str': label_str,
                        '_is_subdivided': True
                    }
                    output_edges_after_subdivision.append(new_sub_edge)
                subdivided_original_indices.add(idx_orig)

    for idx_orig in range(num_edges_initial):
        if idx_orig not in subdivided_original_indices:
            output_edges_after_subdivision.append(current_edges[idx_orig])

    current_edges = output_edges_after_subdivision
    if not current_edges: return []

    adj_info_after_subdivision = build_adjacency_info(current_edges)
    num_edges_after_subdivision = len(current_edges)
    num_nodes_after_subdivision = len(adj_info_after_subdivision['id_to_node'])
    indices_geometrically_deleted = set()

    # 2. Node Deletion
    if random.random() < args.geom_noise_p_delete_node and args.geom_noise_delete_node_ratio > 0 and num_nodes_after_subdivision > 0:
        num_nodes_to_delete_target = int(num_nodes_after_subdivision * args.geom_noise_delete_node_ratio)
        actual_num_nodes_to_delete = min(num_nodes_to_delete_target, num_nodes_after_subdivision)

        if actual_num_nodes_to_delete > 0:
            node_ids_to_delete = random.sample(list(adj_info_after_subdivision['id_to_node'].keys()), k=actual_num_nodes_to_delete)
            for node_id in node_ids_to_delete:
                connected_edge_indices = adj_info_after_subdivision['node_to_edges'].get(node_id, [])
                indices_geometrically_deleted.update(idx for idx in connected_edge_indices if idx < num_edges_after_subdivision)

    # 3. Geometric Edge Deletion
    if random.random() < args.geom_noise_p_delete_edge and args.geom_noise_delete_edge_ratio > 0:
        eligible_indices_for_direct_deletion = [
            i for i in range(num_edges_after_subdivision)
            if i not in indices_geometrically_deleted and not current_edges[i].get('_is_subdivided', False)
        ]
        if eligible_indices_for_direct_deletion:
            num_to_delete_target = int(len(eligible_indices_for_direct_deletion) * args.geom_noise_delete_edge_ratio)
            actual_num_to_delete_this_step = min(num_to_delete_target, len(eligible_indices_for_direct_deletion))
            if actual_num_to_delete_this_step > 0:
                indices_to_delete_this_step = random.sample(eligible_indices_for_direct_deletion, k=actual_num_to_delete_this_step)
                indices_geometrically_deleted.update(indices_to_delete_this_step)

    surviving_edges_after_deletion = [
        edge for i, edge in enumerate(current_edges)
        if i not in indices_geometrically_deleted
    ]
    current_edges = surviving_edges_after_deletion
    if not current_edges: return []

    num_edges_after_deletion = len(current_edges)
    indices_to_process_further = list(range(num_edges_after_deletion))

    # 4. Geometric Breaking (Length and Angle Modification)
    if random.random() < args.geom_noise_p_break_edge and args.geom_noise_break_edge_ratio > 0:
        eligible_indices_for_breaking = [
            i for i in indices_to_process_further
            if not current_edges[i].get('_is_subdivided', False)
        ]
        if eligible_indices_for_breaking:
            num_to_break_target = int(len(eligible_indices_for_breaking) * args.geom_noise_break_edge_ratio)
            actual_num_to_break = min(num_to_break_target, len(eligible_indices_for_breaking))
            if actual_num_to_break > 0:
                indices_to_break = random.sample(eligible_indices_for_breaking, k=actual_num_to_break)
                for idx in indices_to_break:
                    edge = current_edges[idx]
                    if not all(k in edge for k in ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm']): continue
                    x1, y1 = edge['x1_norm'], edge['y1_norm']
                    dx_orig, dy_orig = edge['x2_norm'] - x1, edge['y2_norm'] - y1
                    
                    orig_len = math.sqrt(dx_orig**2 + dy_orig**2)
                    if orig_len < config.COORD_TOLERANCE: continue

                    new_length = orig_len * args.geom_noise_break_length_factor
                    edge['x2_norm'] = x1 + new_length
                    edge['y2_norm'] = y1
                    edge['_was_broken'] = True

    # 5. Angle and Length Noise (Gaussian Noise)
    apply_angle_gauss_noise = random.random() < args.geom_noise_p_angle_noise and args.geom_noise_angle_std > 0
    apply_length_gauss_noise = random.random() < args.geom_noise_p_length_noise and args.geom_noise_length_std > 0

    if apply_angle_gauss_noise or apply_length_gauss_noise:
        for idx in indices_to_process_further:
            edge = current_edges[idx]
            if edge.get('_is_subdivided', False) or edge.get('_was_broken', False):
                continue
            if not all(k in edge for k in ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm']): continue

            x1, y1, x2, y2 = edge['x1_norm'], edge['y1_norm'], edge['x2_norm'], edge['y2_norm']
            dx, dy = x2 - x1, y2 - y1
            current_length = math.sqrt(dx**2 + dy**2)
            current_angle = math.atan2(dy, dx)

            noisy_length, noisy_angle = current_length, current_angle

            if apply_angle_gauss_noise:
                noisy_angle += random.gauss(0, args.geom_noise_angle_std)
                noisy_angle = math.remainder(noisy_angle + math.pi, 2 * math.pi) - math.pi

            if apply_length_gauss_noise and current_length > config.COORD_TOLERANCE:
                noise_val = random.gauss(0, current_length * args.geom_noise_length_std)
                noisy_length += noise_val
                noisy_length = max(0.0, noisy_length)

            if abs(noisy_angle - current_angle) > 1e-9 or abs(noisy_length - current_length) > 1e-9:
                 edge['x2_norm'] = x1 + noisy_length * math.cos(noisy_angle)
                 edge['y2_norm'] = y1 + noisy_length * math.sin(noisy_angle)

    # 6. Coordinate Noise (Gaussian Jitter on Endpoints)
    if random.random() < args.geom_noise_p_coord_noise and args.geom_noise_coord_std > 0:
        coord_std = args.geom_noise_coord_std
        for idx in indices_to_process_further:
            edge = current_edges[idx]
            if 'x1_norm' in edge: edge['x1_norm'] += random.gauss(0, coord_std)
            if 'y1_norm' in edge: edge['y1_norm'] += random.gauss(0, coord_std)
            if 'x2_norm' in edge: edge['x2_norm'] += random.gauss(0, coord_std)
            if 'y2_norm' in edge: edge['y2_norm'] += random.gauss(0, coord_std)

    return current_edges


# Collate Function
def graph_aware_collate_fn_with_augmentation(batch, scaler, label_encoder, args):
    """
    Collates a batch of graph data, applies augmentation, and pads sequences.
    This function is CPU-bound and relies on DataLoader's multi-processing for speed.
    The per-graph loop is necessary due to variable sequence lengths and complex augmentations.
    """
    processed_features_list = []
    processed_labels_list = []

    for graph_edges_raw_unfiltered in batch:
        graph_edges_raw = [
            edge for edge in graph_edges_raw_unfiltered
            if isinstance(edge, dict) and all(k in edge for k in ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm', 'label_str'])
        ]
        if not graph_edges_raw:
            processed_features_list.append(torch.empty((0, len(config.FEATURE_COLS)), dtype=torch.float))
            processed_labels_list.append(torch.empty((0,), dtype=torch.long))
            continue

        adj_info = build_adjacency_info(graph_edges_raw)

        graph_edges_after_noise = apply_online_geometric_noise(graph_edges_raw, adj_info, args)

        if not graph_edges_after_noise:
            processed_features_list.append(torch.empty((0, len(config.FEATURE_COLS)), dtype=torch.float))
            processed_labels_list.append(torch.empty((0,), dtype=torch.long))
            continue

        graph_features_dict_list_final = []
        graph_labels_str_list_final = []

        for edge_dict_processed in graph_edges_after_noise:
            try:
                if not all(k in edge_dict_processed for k in ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm', 'label_str']):
                    continue

                features_calculated = calculate_edge_features(
                    edge_dict_processed['x1_norm'], edge_dict_processed['y1_norm'],
                    edge_dict_processed['x2_norm'], edge_dict_processed['y2_norm']
                )
                graph_features_dict_list_final.append(features_calculated)
                graph_labels_str_list_final.append(edge_dict_processed['label_str'])
            except Exception:
                continue

        if not graph_features_dict_list_final:
            processed_features_list.append(torch.empty((0, len(config.FEATURE_COLS)), dtype=torch.float))
            processed_labels_list.append(torch.empty((0,), dtype=torch.long))
            continue

        feature_values_ordered = [
            [feat_d.get(key, 0.0) for key in config.FEATURE_COLS]
            for feat_d in graph_features_dict_list_final
        ]
        features_np = np.array(feature_values_ordered, dtype=np.float32)

        try:
            scaled_features_np = scaler.transform(features_np) if features_np.shape[0] > 0 else np.empty((0, features_np.shape[1]), dtype=np.float32)
        except Exception:
            scaled_features_np = features_np

        try:
            known_labels_mask = np.isin(graph_labels_str_list_final, label_encoder.classes_)
            labels_to_encode_np = np.array(graph_labels_str_list_final)[known_labels_mask]

            scaled_features_np_aligned = scaled_features_np[known_labels_mask]

            if labels_to_encode_np.size > 0:
                encoded_labels_np = label_encoder.transform(labels_to_encode_np)
            else:
                encoded_labels_np = np.empty((0,), dtype=int)

            if scaled_features_np_aligned.shape[0] == 0:
                processed_features_list.append(torch.empty((0, len(config.FEATURE_COLS)), dtype=torch.float))
                processed_labels_list.append(torch.empty((0,), dtype=torch.long))
                continue
        except Exception:
            processed_features_list.append(torch.empty((0, len(config.FEATURE_COLS)), dtype=torch.float))
            processed_labels_list.append(torch.empty((0,), dtype=torch.long))
            continue

        features_tensor = torch.from_numpy(scaled_features_np_aligned).float()
        labels_tensor = torch.from_numpy(encoded_labels_np).long()

        processed_features_list.append(features_tensor)
        processed_labels_list.append(labels_tensor)

    if not processed_features_list:
        return torch.empty((0, 0, len(config.FEATURE_COLS))), torch.empty((0, 0)), torch.empty((0, 0), dtype=torch.bool)

    features_padded = rnn_utils.pad_sequence(processed_features_list, batch_first=True, padding_value=PAD_VALUE_FEATURES)
    labels_padded = rnn_utils.pad_sequence(processed_labels_list, batch_first=True, padding_value=PAD_VALUE_LABELS)

    lengths = torch.tensor([f.shape[0] for f in processed_features_list], dtype=torch.long)
    attention_mask = torch.arange(features_padded.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)

    return features_padded, labels_padded, attention_mask


# Loading Functions
def load_scaler_encoder(scaler_path, encoder_path):
    try:
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        with open(encoder_path, 'rb') as f: label_encoder = pickle.load(f)
        num_classes = len(label_encoder.classes_)
        print(f"Loaded Scaler from {scaler_path}")
        print(f"Loaded LabelEncoder from {encoder_path}. Classes: {label_encoder.classes_} ({num_classes} classes)")
        return scaler, label_encoder, num_classes
    except FileNotFoundError as e:
        print(f"Error loading scaler/encoder: {e}. Check paths.")
        raise
    except Exception as e:
        print(f"Unexpected error loading scaler/encoder: {e}")
        traceback.print_exc()
        raise


def load_and_split_data(args, scaler, label_encoder):
    data_path = args.train_val_data_json_path
    val_split = args.val_split
    batch_size = args.batch_size
    num_classes = len(label_encoder.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_class_weight = args.max_class_weight

    print(f"Loading graph IDs and preparing splits from {data_path}...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f: all_graph_data_for_ids = json.load(f)
        all_graph_ids = sorted(all_graph_data_for_ids.keys())
        del all_graph_data_for_ids
        total_size = len(all_graph_ids)
        if total_size == 0: raise ValueError("No graphs found in the JSON data file.")

        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        if train_size <= 0 or (val_split > 0 and val_size <= 0):
            raise ValueError(f"Invalid split sizes: train_size={train_size}, val_size={val_size} from total={total_size} and val_split={val_split}")

        shuffled_ids = random.sample(all_graph_ids, k=total_size)
        train_graph_ids, val_graph_ids = shuffled_ids[:train_size], shuffled_ids[train_size:]
        print(f"Total graphs: {total_size}. Train IDs: {len(train_graph_ids)}, Val IDs: {len(val_graph_ids)}")

        train_dataset = GraphSequenceDataset(data_path=data_path, graph_ids=train_graph_ids)
        val_dataset = GraphSequenceDataset(data_path=data_path, graph_ids=val_graph_ids) if val_size > 0 else None

    except Exception as e:
        print(f"FATAL: Error loading/splitting data from {data_path}: {e}")
        traceback.print_exc()
        raise IOError(f"Error in load_and_split_data: {e}")

    print("Calculating class weights from training subset...")
    all_train_labels_str = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f: source_data_for_weights = json.load(f)
        for graph_id in tqdm(train_graph_ids, desc="Extracting Labels for Weights", leave=False, ncols=80):
             graph_edges = source_data_for_weights.get(graph_id, [])
             all_train_labels_str.extend([edge['label_str'] for edge in graph_edges if isinstance(edge, dict) and 'label_str' in edge])
        del source_data_for_weights
    except Exception as e:
        print(f"Warning: Error reading data for class weight calculation: {e}. Using uniform weights.")
        all_train_labels_str = []

    if not all_train_labels_str:
        print("Warning: No labels found for training set during weight calculation. Using uniform log weights.")
        class_weights_tensor = torch.full((num_classes,), math.log1p(1.0), dtype=torch.float).to(device)
    else:
        try:
            from sklearn.utils.class_weight import compute_class_weight
            known_labels_mask_for_weights = np.isin(all_train_labels_str, label_encoder.classes_)
            labels_for_weights_calc = np.array(all_train_labels_str)[known_labels_mask_for_weights]

            if labels_for_weights_calc.size == 0:
                raise ValueError("No known labels found in training set after filtering for weight calculation.")

            encoded_labels_for_weights = label_encoder.transform(labels_for_weights_calc)
            raw_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=encoded_labels_for_weights)

            processed_weights = np.log1p(raw_weights)
            weight_type_str = "Logarithmic (log1p)"
            if max_class_weight is not None:
                processed_weights = np.clip(processed_weights, a_min=None, a_max=max_class_weight)
                weight_type_str += f" clipped at {max_class_weight}"
            class_weights_tensor = torch.tensor(processed_weights, dtype=torch.float).to(device)
            print(f"Using {weight_type_str} class weights.")
        except Exception as e:
            print(f"Warning: Class weight calculation failed ({e}). Using uniform log weights.")
            traceback.print_exc()
            class_weights_tensor = torch.full((num_classes,), math.log1p(1.0), dtype=torch.float).to(device)

    collate_fn_train_partial = functools.partial(
        graph_aware_collate_fn_with_augmentation,
        scaler=scaler,
        label_encoder=label_encoder,
        args=args
    )

    args_no_geom_noise_for_val = argparse.Namespace(**vars(args))
    args_no_geom_noise_for_val.apply_geom_noise = False

    collate_fn_eval_partial = functools.partial(
        graph_aware_collate_fn_with_augmentation,
        scaler=scaler,
        label_encoder=label_encoder,
        args=args_no_geom_noise_for_val
    )

    num_workers = args.num_workers
    pin_memory_setting = True if device.type == 'cuda' else False
    print(f"DataLoader: num_workers={num_workers}, pin_memory={pin_memory_setting}, persistent_workers=True")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory_setting,
        collate_fn=collate_fn_train_partial,
        persistent_workers=True if num_workers > 0 else False, drop_last=False
    ) if train_dataset else None

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory_setting,
        collate_fn=collate_fn_eval_partial,
        persistent_workers=True if num_workers > 0 else False
    ) if val_dataset else None

    if train_loader is None: print("Warning: Training DataLoader creation failed (train_dataset might be None or empty).")
    if val_loader is None and val_split > 0: print("Warning: Validation DataLoader not created (val_dataset might be None or empty, or val_split is 0).")

    return train_loader, val_loader, class_weights_tensor


def load_test_data(args, scaler, label_encoder, apply_noise=False):
    """
    Loads test data. The 'apply_noise' parameter controls if geometric noise is enabled.
    """
    data_path = args.eval_data_json_path
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading test graph data from {data_path} (Requested noise application for this loader: {apply_noise})...")
    test_loader = None
    try:
        with open(data_path, 'r', encoding='utf-8') as f: all_graph_data_for_ids = json.load(f)
        test_graph_ids = sorted(all_graph_data_for_ids.keys())
        del all_graph_data_for_ids
        if not test_graph_ids:
            print(f"Warning: No graph IDs found for test set in '{data_path}'.")
            return None
        test_dataset = GraphSequenceDataset(data_path=data_path, graph_ids=test_graph_ids)
        if len(test_dataset) == 0:
            print(f"Warning: Test dataset from '{data_path}' is empty after loading IDs.")
            return None

        args_for_collate = argparse.Namespace(**vars(args))

        if apply_noise:
            args_for_collate.apply_geom_noise = True
            print("  Note: Collate function for this loader WILL apply geometric noise if its global probability is met (using parameters from main args).")
        else:
            args_for_collate.apply_geom_noise = False
            print("  Note: Collate function for this loader will NOT apply geometric or feature noise.")

        collate_fn_test_partial = functools.partial(
            graph_aware_collate_fn_with_augmentation,
            scaler=scaler,
            label_encoder=label_encoder,
            args=args_for_collate
        )

        num_workers = args.num_workers
        pin_memory_setting = True if device.type == 'cuda' else False

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=pin_memory_setting,
            collate_fn=collate_fn_test_partial,
            persistent_workers=True if num_workers > 0 else False
        )
        print(f"Test DataLoader created with {len(test_dataset)} graphs.")
    except FileNotFoundError:
        print(f"FATAL: Test data file not found: '{data_path}'.")
        return None
    except Exception as e:
        print(f"FATAL: Error processing test data from '{data_path}': {e}.")
        traceback.print_exc()
        return None
    return test_loader
