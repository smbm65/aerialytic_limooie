# refiner_config.py
import argparse
import os
import re
import sys
import traceback
import math

# Base Geometric Feature Columns (Must match the original model's feature calculation)
GEOM_FEATURE_COLS = [ # Ensure this list is comprehensive and matches calculate_geometric_features_for_refiner
    'length', 'sin_angle', 'cos_angle', 'x1_norm', 'y1_norm', 'x2_norm', 'y2_norm',
    'mid_x_norm', 'mid_y_norm', 'abs_cos_angle', 'abs_sin_angle', 'angle_rad',
    'is_horizontal', 'is_vertical', 'is_positive_slope', 'is_negative_slope',
    'cross_vec_x', 'cross_vec_y', 'norm_sum_vec_x', 'norm_sum_vec_y',
    'norm45_x', 'norm45_y', 'norm135_x', 'norm135_y',
]
OFFSET_POINTS_REF = ['start', 'mid', 'end']
OFFSET_ANGLES_REF = ['0', '45', '90', '135', '180', '225', '270', '315']
OFFSET_VECTOR_FEATURES_REF = []
for point in OFFSET_POINTS_REF:
    for angle in OFFSET_ANGLES_REF:
        OFFSET_VECTOR_FEATURES_REF.append(f'{point}_{angle}_x')
        OFFSET_VECTOR_FEATURES_REF.append(f'{point}_{angle}_y')
GEOM_FEATURE_COLS += OFFSET_VECTOR_FEATURES_REF


PROB_FEATURE_COLS_TEMPLATE = "orig_prob_class_{}"
ONEHOT_FEATURE_COLS_TEMPLATE = "orig_onehot_class_{}"
PROB_FEATURE_COLS = []
ONEHOT_FEATURE_COLS = []
ALL_FEATURE_COLS = []

TARGET_COL_REF = 'label_encoded'
LABEL_COL_STR_REF = 'label_str'
ORIG_PRED_IDX_COL_REF = 'original_pred_label_idx'
CORRECTNESS_TARGET_COL_REF = 'is_original_pred_correct'
COORD_TOLERANCE_REF = 1e-6
NODE_COORD_PRECISION_REF = 5
ORIENT_TOLERANCE_REF = 5e-2
S_45_REF = 1.0 / math.sqrt(2.0)
OFFSET_DIRECTIONS_REF = {
    '0': (1.0, 0.0), '45': (S_45_REF, S_45_REF), '90': (0.0, 1.0), '135': (-S_45_REF, S_45_REF),
    '180': (-1.0, 0.0), '225': (-S_45_REF, -S_45_REF), '270': (0.0, -1.0), '315': (S_45_REF, -S_45_REF),
}


def calculate_geometric_features_for_refiner(x1n, y1n, x2n, y2n):
    dx = x2n - x1n
    dy = y2n - y1n
    length = math.sqrt(dx**2 + dy**2)
    eps = 1e-9

    is_horizontal, is_vertical, is_positive_slope, is_negative_slope = 0.0, 0.0, 0.0, 0.0
    angle_rad = 0.0
    cos_angle, sin_angle = 1.0, 0.0

    if length > COORD_TOLERANCE_REF:
        angle_rad = math.atan2(dy, dx)
        cos_angle = dx / length
        sin_angle = dy / length
        if abs(sin_angle) < ORIENT_TOLERANCE_REF: is_horizontal = 1.0
        elif abs(cos_angle) < ORIENT_TOLERANCE_REF: is_vertical = 1.0
        else:
            is_positive_slope = 1.0 if (dx * dy) > 0 else 0.0
            is_negative_slope = 1.0 if (dx * dy) < 0 else 0.0
    else: # If length is very small, treat as horizontal to avoid division by zero in angle calcs
        is_horizontal = 1.0
        cos_angle = 1.0 # Default orientation
        sin_angle = 0.0

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

    s_45 = S_45_REF
    norm45_x = s_45 * (cos_angle - sin_angle)
    norm45_y = s_45 * (cos_angle + sin_angle)
    norm135_x = s_45 * (-cos_angle + sin_angle) 
    norm135_y = s_45 * ( cos_angle + sin_angle)

    geom_features_dict = {
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

    points = {'start': (x1n, y1n), 'mid': (mid_x_norm, mid_y_norm), 'end': (x2n, y2n)}
    for point_name, (px, py) in points.items():
        for angle_name, (dx_off, dy_off) in OFFSET_DIRECTIONS_REF.items():
            geom_features_dict[f'{point_name}_{angle_name}_x'] = dx_off
            geom_features_dict[f'{point_name}_{angle_name}_y'] = dy_off

    return {key: geom_features_dict.get(key, 0.0) for key in GEOM_FEATURE_COLS}


def update_feature_columns(num_classes):
    global PROB_FEATURE_COLS, ONEHOT_FEATURE_COLS, ALL_FEATURE_COLS, GEOM_FEATURE_COLS
    PROB_FEATURE_COLS = [PROB_FEATURE_COLS_TEMPLATE.format(i) for i in range(num_classes)]
    ONEHOT_FEATURE_COLS = [ONEHOT_FEATURE_COLS_TEMPLATE.format(i) for i in range(num_classes)]
    ALL_FEATURE_COLS = GEOM_FEATURE_COLS + PROB_FEATURE_COLS + ONEHOT_FEATURE_COLS
    return ALL_FEATURE_COLS


def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Refiner Model for Graph Edge Classification')
    group_common = parser.add_argument_group('Refiner Common Paths and Model Structure')
    group_common.add_argument('--base_dir', type=str, default='.', help='Base directory for refiner outputs')
    group_common.add_argument('--dataset_dir', type=str, default="refiner_dataset", help='Subdirectory for refiner preprocessed data')
    group_common.add_argument('--checkpoint_dir', type=str, default="refiner_cpk", help='Base subdirectory for refiner checkpoints')
    group_common.add_argument('--plot_dir', type=str, default="refiner_plots", help='Base subdirectory for refiner plots')
    group_common.add_argument('--run_output_dir', type=str, default=None, help='Specific run output directory for refiner')
    group_common.add_argument('--d_model', type=int, default=256)
    group_common.add_argument('--nhead', type=int, default=8)
    group_common.add_argument('--num_layers', type=int, default=6)
    group_common.add_argument('--dim_ff', type=int, default=1024)
    group_common.add_argument('--dropout', type=float, default=0.1)
    group_common.add_argument('--k_nearest', type=int, default=16)
    group_common.add_argument('--knn_distance_threshold', type=float, default=1.0)
    group_common.add_argument('--norm_first', action='store_true')
    group_common.add_argument('--num_workers', type=int, default=8, help='DataLoader workers (set to 0 for debugging)')

    group_refiner_spec = parser.add_argument_group('Refiner Specific Parameters')
    group_refiner_spec.add_argument('--refiner_input_feature_mode', type=str, default="onehot_only", choices=["probs_and_onehot", "probs_only", "onehot_only", "geom_only"])
    group_refiner_spec.add_argument('--label_corruption_frac_graphs', type=float, default=0.75)
    group_refiner_spec.add_argument('--label_corruption_frac_edges', type=float, default=0.1)
    group_refiner_spec.add_argument('--main_loss_weight', type=float, default=2.0)
    group_refiner_spec.add_argument('--correctness_loss_weight', type=float, default=1.0)
    group_refiner_spec.add_argument('--refiner_main_head_dims', type=int, nargs='+', default=[128, 32], help='List of hidden layer sizes for the main MLP classification head.')
    group_refiner_spec.add_argument('--refiner_correctness_head_dims', type=int, nargs='+', default=[32], help='List of hidden layer sizes for the correctness MLP head.')

    group_train = parser.add_argument_group('Refiner Training Arguments')
    group_train.add_argument('--train_val_data_json_base_name', type=str, default="refiner_geom_Train_RefinerData", help='Base name for Refiner Train/Val JSON Manifest')
    group_train.add_argument('--scaler_base_name', type=str, default="refiner_scaler_geom_only", help='Base name for refiner scaler')
    group_train.add_argument('--encoder_base_name', type=str, default="refiner_encoder", help='Base name for refiner encoder')
    group_train.add_argument('--best_model_base_name', type=str, default="refiner_best_model")
    group_train.add_argument('--latest_ckpt_base_name', type=str, default="refiner_latest_ckpt")
    group_train.add_argument('--plot_base_name_train', type=str, default="refiner_cm_train")
    group_train.add_argument('--plot_base_name_convergence', type=str, default="refiner_convergence")
    group_train.add_argument('--resume_from', type=str, default=None)
    group_train.add_argument('--batch_size', type=int, default=128)
    group_train.add_argument('--epochs', type=int, default=100)
    group_train.add_argument('--lr', type=float, default=1e-4)
    group_train.add_argument('--val_split', type=float, default=0.15)
    group_train.add_argument('--lr_patience', type=int, default=5)
    group_train.add_argument('--lr_factor', type=float, default=0.1)
    group_train.add_argument('--early_stop', type=int, default=10)
    group_train.add_argument('--max_class_weight', type=float, default=None)
    group_train.add_argument('--correctness_loss_weighting_mode', type=str, default='manual',
                             choices=['none', 'linear', 'log', 'manual'],
                             help="Method to weight the correctness head loss to balance classes. 'none' is default.")
    group_train.add_argument('--correctness_loss_manual_pos_weight', type=float, default=0.5,
                             help="Manual positive class weight for correctness loss (only if mode is 'manual').")
    group_train.add_argument('--max_correctness_pos_weight', type=float, default=10.0,
                             help="Maximum value to clip the positive class weight for the correctness loss (for 'linear' and 'log' modes).")

    group_geom_noise = parser.add_argument_group('Refiner Online GEOMETRIC Noise Augmentation')
    group_geom_noise.add_argument('--apply_geom_noise', action='store_true')
    group_geom_noise.add_argument('--geom_noise_global_p', type=float, default=0.7)
    group_geom_noise.add_argument('--geom_noise_p_coord_noise', type=float, default=0.02)
    group_geom_noise.add_argument('--geom_noise_coord_std', type=float, default=0.02)

    group_eval = parser.add_argument_group('Refiner Evaluation Arguments')
    group_eval.add_argument('--eval_data_json_base_name', type=str, default="refiner_geom_Test_RefinerData", help='Base name for Refiner Test JSON Manifest')
    group_eval.add_argument('--eval_model_path', type=str, default=None)
    group_eval.add_argument('--eval_plot_base_name', type=str, default="refiner_cm_eval")
    group_eval.add_argument('--eval_plot_path', type=str, default=None)
    group_eval.add_argument('--eval_with_geom_noise', action='store_true')
    group_eval.add_argument('--eval_cm_data_base_name', type=str, default="refiner_cm_data_eval")

    group_viz = parser.add_argument_group('Refiner Visualization Arguments')
    group_viz.add_argument('--viz_json_base_name', type=str, default="refiner_geom_Test_RefinerData", help='Base name for Refiner Viz JSON Manifest (usually Test set)')
    group_viz.add_argument('--viz_graph_ids', type=str, nargs='+', default=None)
    group_viz.add_argument('--viz_model_path', type=str, default=None)
    group_viz.add_argument('--viz_output_plot_base_name', type=str, default="refiner_viz_pred")
    group_viz.add_argument('--viz_output_plot_path', type=str, default=None)
    group_viz.add_argument('--viz_show_endpoints', action='store_true')
    group_viz.add_argument('--viz_corrupt_labels', action='store_true')
    group_viz.add_argument('--viz_force_delete_edge', action='store_true')
    group_viz.add_argument('--viz_force_break_edge', action='store_true')
    group_viz.add_argument('--viz_force_angle_noise', action='store_true')
    group_viz.add_argument('--viz_force_length_noise', action='store_true')
    group_viz.add_argument('--viz_force_delete_node', action='store_true')
    group_viz.add_argument('--viz_force_subdivide_edge', action='store_true')
    group_viz.add_argument('--viz_force_coord_noise', action='store_true')

    group_prep = parser.add_argument_group('Refiner Preprocessing Arguments')
    group_prep.add_argument('--prep_input_dir_orig_model_output', type=str, default="prediction_json_outputs/clean_output/train")
    group_prep.add_argument('--prep_input_dir_orig_model_output_test', type=str, default="prediction_json_outputs/clean_output/test")
    group_prep.add_argument('--prep_json_pattern_orig_model_output', type=str, default="*Graph_Probs.json")
    group_prep.add_argument('--prep_output_dir', type=str, default="refiner_dataset")
    group_prep.add_argument('--prep_output_suffix_train', type=str, default="_Train")
    group_prep.add_argument('--prep_output_suffix_test', type=str, default="_Test")
    group_prep.add_argument('--num_classes_orig_model', type=int, default=6)

    try:
        args = parser.parse_args()
    except SystemExit as e:
         print(f"Argument parsing error: {e}")
         parser.print_help()
         sys.exit(1)

    try:
        lr_str = f"{args.lr:.0e}".replace('-', 'neg')
        do_str = str(args.dropout).replace('.', 'p')
        bs_str = str(args.batch_size)
        geom_noise_tag = "GeomNoiseOn" if args.apply_geom_noise else "GeomNoiseOff"
        label_corr_tag = "LblCorrOn" if args.label_corruption_frac_graphs > 0 and args.label_corruption_frac_edges > 0 else "LblCorrOff"
        ref_feat_mode_short = {'probs_and_onehot': 'PaO', 'probs_only': 'PrO', 'onehot_only': 'OhO', 'geom_only': 'GeoO'}.get(args.refiner_input_feature_mode, 'Cust')

        hparam_str = f"Ref_lr{lr_str}_bs{bs_str}_do{do_str}_{ref_feat_mode_short}_{geom_noise_tag}_{label_corr_tag}"
        if args.k_nearest: hparam_str += f"_knn{args.k_nearest}"
        if args.knn_distance_threshold is not None: hparam_str += f"_kthr{str(args.knn_distance_threshold).replace('.', 'p')}"
        hparam_str += f"_d{args.d_model}nh{args.nhead}nl{args.num_layers}"
        
        if args.refiner_main_head_dims:
            hparam_str += f"_mMLP{'x'.join(map(str, args.refiner_main_head_dims))}"
        if args.refiner_correctness_head_dims:
            hparam_str += f"_cMLP{'x'.join(map(str, args.refiner_correctness_head_dims))}"

        hparam_str = re.sub(r'[<>:"/\\|?*]+', '_', hparam_str) # Sanitize for path
        args.hparam_str = hparam_str
    except Exception as e:
        print(f"Error generating refiner hparam string: {e}")
        traceback.print_exc()
        args.hparam_str = "default_refiner_hparams"

    try:
        args.prep_output_dir = os.path.join(args.base_dir, args.prep_output_dir)
        args.dataset_dir = args.prep_output_dir
        hparam_subdir = args.hparam_str
        if args.run_output_dir:
            if not os.path.isabs(args.run_output_dir):
                args.run_output_dir = os.path.join(args.base_dir, args.run_output_dir)
            output_parent_for_hparam_subdir = args.run_output_dir
        else:
             output_parent_for_hparam_subdir = os.path.join(args.base_dir, args.checkpoint_dir)
        args.current_run_dir = os.path.join(output_parent_for_hparam_subdir, hparam_subdir)
        args.current_plot_dir = args.current_run_dir 
        args.best_model_path = os.path.join(args.current_run_dir, f"{args.best_model_base_name}.pt")
        args.latest_ckpt_path = os.path.join(args.current_run_dir, f"{args.latest_ckpt_base_name}.pt")
        args.plot_path_train = os.path.join(args.current_plot_dir, f"{args.plot_base_name_train}.png")
        args.plot_path_convergence = os.path.join(args.current_plot_dir, f"{args.plot_base_name_convergence}.png")
        args.train_val_data_json_path = os.path.join(args.dataset_dir, f"{args.train_val_data_json_base_name}_manifest.json")
        args.eval_data_json_path = os.path.join(args.dataset_dir, f"{args.eval_data_json_base_name}_manifest.json")
        args.viz_json_path = os.path.join(args.dataset_dir, f"{args.viz_json_base_name}_manifest.json")
        args.scaler_path = os.path.join(args.dataset_dir, f"{args.scaler_base_name}_scaler.pkl")
        args.encoder_path = os.path.join(args.dataset_dir, f"{args.encoder_base_name}_label_encoder.pkl")
        if args.resume_from and not os.path.isabs(args.resume_from):
            if not os.path.dirname(args.resume_from):
                 args.resume_from = os.path.join(args.current_run_dir, args.resume_from)
        args.eval_scaler_path = args.scaler_path
        args.eval_encoder_path = args.encoder_path
        if args.eval_model_path is None: args.eval_model_path = args.best_model_path
        elif not os.path.isabs(args.eval_model_path) and not os.path.dirname(args.eval_model_path):
             args.eval_model_path = os.path.join(args.current_run_dir, args.eval_model_path)
        default_eval_plot_path = os.path.join(args.current_plot_dir, f"{args.eval_plot_base_name}.png")
        if args.eval_plot_path is None: args.eval_plot_path = default_eval_plot_path
        elif not os.path.isabs(args.eval_plot_path) and not os.path.dirname(args.eval_plot_path):
            args.eval_plot_path = os.path.join(args.current_plot_dir, args.eval_plot_path)
        args.eval_cm_data_path = args.eval_plot_path.replace(".png", "_cm_data.npy")
        args.viz_scaler_path = args.scaler_path
        args.viz_encoder_path = args.encoder_path
        if args.viz_model_path is None: args.viz_model_path = args.best_model_path
        elif not os.path.isabs(args.viz_model_path) and not os.path.dirname(args.viz_model_path):
            args.viz_model_path = os.path.join(args.current_run_dir, args.viz_model_path)
        default_viz_plot_pattern = os.path.join(args.current_plot_dir, f"{args.viz_output_plot_base_name}_{{graph_id}}.png")
        if args.viz_output_plot_path is None: args.viz_output_plot_pattern = default_viz_plot_pattern
        elif '{graph_id}' not in args.viz_output_plot_path:
            viz_plot_custom_dir = args.viz_output_plot_path
            if not os.path.isabs(viz_plot_custom_dir):
                viz_plot_custom_dir = os.path.join(args.base_dir, viz_plot_custom_dir)
            args.viz_output_plot_pattern = os.path.join(viz_plot_custom_dir, f"{args.viz_output_plot_base_name}_{{graph_id}}.png")
        else:
            args.viz_output_plot_pattern = args.viz_output_plot_path
    
    except Exception as e:
         print(f"Error constructing refiner file paths: {e}")
         traceback.print_exc()
         sys.exit(1)

    return args


def ensure_dirs(args):
    try:
        os.makedirs(args.prep_output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.prep_output_dir, "graph_data"), exist_ok=True)
        os.makedirs(args.current_run_dir, exist_ok=True)
        os.makedirs(args.current_plot_dir, exist_ok=True)
        for plot_path_attr_val in [args.plot_path_train, args.plot_path_convergence, args.eval_plot_path]:
            if plot_path_attr_val:
                plot_dir_val = os.path.dirname(plot_path_attr_val)
                if plot_dir_val: os.makedirs(plot_dir_val, exist_ok=True)
        if args.eval_with_geom_noise and hasattr(args, 'eval_plot_path') and args.eval_plot_path:
            noisy_plot_path = get_noisy_eval_plot_path(args, for_refiner=True)
            noisy_plot_dir = os.path.dirname(noisy_plot_path)
            if noisy_plot_dir: os.makedirs(noisy_plot_dir, exist_ok=True)
        if hasattr(args, 'viz_output_plot_pattern') and args.viz_output_plot_pattern:
             viz_plot_base_dir = os.path.dirname(args.viz_output_plot_pattern.replace("{graph_id}", "dummy_id_for_path"))
             if viz_plot_base_dir : os.makedirs(viz_plot_base_dir, exist_ok=True)
    except OSError as e: print(f"Error creating refiner directories: {e}")


def get_noisy_eval_plot_path(args, for_refiner=True):
    base_path = args.eval_plot_path
    if not base_path:
        base_path = os.path.join(args.current_plot_dir, f"{args.eval_plot_base_name}.png")
        print(f"Warning: eval_plot_path not defined, using default for noisy path: {base_path}")
    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    noisy_filename = f"{name}_geom_noisy{ext}"
    return os.path.join(directory, noisy_filename)

args_parsed = None
