# config.py
import argparse
import os
import re
import sys
import traceback
import math

# Feature Columns (72 features)
# Base geometric features
base_geom_features = [
    'length', 'sin_angle', 'cos_angle', 'x1_norm', 'y1_norm', 'x2_norm', 'y2_norm',
    'mid_x_norm', 'mid_y_norm', 'abs_cos_angle', 'abs_sin_angle',
    'angle_rad',
]
flag_features = [
    'is_horizontal', 'is_vertical', 'is_positive_slope', 'is_negative_slope',
]
derived_vector_features = [
    'cross_vec_x', 'cross_vec_y', # Norm90
    'norm_sum_vec_x', 'norm_sum_vec_y',
    'norm45_x', 'norm45_y',   # Norm45
    'norm135_x', 'norm135_y', # Norm135
]
offset_points = ['start', 'mid', 'end']
offset_angles = ['0', '45', '90', '135', '180', '225', '270', '315']
offset_vector_features = []

for point in offset_points:
    for angle in offset_angles:
        offset_vector_features.append(f'{point}_{angle}_x')
        offset_vector_features.append(f'{point}_{angle}_y')

FEATURE_COLS = (
    base_geom_features
    + flag_features
    + derived_vector_features
    + offset_vector_features
)

# Indices constants
try:
    X1_IDX = FEATURE_COLS.index('x1_norm')
    Y1_IDX = FEATURE_COLS.index('y1_norm')
    X2_IDX = FEATURE_COLS.index('x2_norm')
    Y2_IDX = FEATURE_COLS.index('y2_norm')
    MID_X_IDX = FEATURE_COLS.index('mid_x_norm')
    MID_Y_IDX = FEATURE_COLS.index('mid_y_norm')
    if 'start_0_x' not in FEATURE_COLS:
        print("Warning: Offset features like 'start_0_x' might not be in FEATURE_COLS.")
except ValueError as e:
    print(f"FATAL ERROR: Essential coordinate feature not found in FEATURE_COLS: {e}")
    sys.exit(1)

TARGET_COL = 'label_encoded'
LABEL_COL_STR = 'label_str'
COORD_TOLERANCE = 1e-6
NODE_COORD_PRECISION = 5
ORIENT_TOLERANCE = 5e-2

s_45 = 1.0 / math.sqrt(2.0) # Precompute for preprocess.py

OFFSET_DIRECTIONS = { # Used in preprocess.py
    '0':   (1.0, 0.0), '45':  (s_45, s_45), '90':  (0.0, 1.0), '135': (-s_45, s_45),
    '180': (-1.0, 0.0), '225': (-s_45, -s_45), '270': (0.0, -1.0), '315': (s_45, -s_45),
}

MIN_SCALE_PREVENTION = 1e-9 # For inference script

# FUNCTION to define all arguments and return the parser object
def get_configured_parser():
    parser = argparse.ArgumentParser(description='Transformer for Graph Edge Sequence Classification & Visualization', add_help=False)

    # Common Paths and Model Structure
    group_common = parser.add_argument_group('Common Paths and Model Structure')
    group_common.add_argument('--base_dir', type=str, default='.', help='Base directory')
    group_common.add_argument('--dataset_dir', type=str, default="dataset", help='Data subdirectory')
    group_common.add_argument('--checkpoint_dir', type=str, default="cpk", help='Checkpoints subdirectory')
    group_common.add_argument('--plot_dir', type=str, default="plots", help='Plots subdirectory')
    group_common.add_argument('--run_output_dir', type=str, default=None, help='Specific run output directory (overrides cpk/plots for current run)')
    group_common.add_argument('--scaler_base_name', type=str, default="bboxnorm_orient_Train_GraphSeq", help='Base name for scaler')
    group_common.add_argument('--encoder_base_name', type=str, default="bboxnorm_orient_Train_GraphSeq", help='Base name for encoder')
    group_common.add_argument('--train_val_data_json_base_name', type=str, default="visualization_data_bboxnorm_orient_Train_GraphSeq", help='Base name for Train/Validation JSON')
    group_common.add_argument('--eval_data_json_base_name', type=str, default="visualization_data_bboxnorm_orient_Test_GraphSeq", help='Base name for Eval JSON')
    group_common.add_argument('--viz_json_base_name', type=str, default="visualization_data_bboxnorm_orient_Test_GraphSeq", help='Base name for Viz JSON')
    group_common.add_argument('--d_model', type=int, default=256, help='Transformer model dimension')
    group_common.add_argument('--nhead', type=int, default=8, help='Number of attention heads (must divide d_model)')
    group_common.add_argument('--num_layers', type=int, default=6, help='Number of Transformer encoder layers')
    group_common.add_argument('--dim_ff', type=int, default=1024, help='Dimension of the feedforward network')
    group_common.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    group_common.add_argument('--k_nearest', type=int, default=16, help='K for K-Nearest Lines attention. Attend to only the k closest lines. If None, use full attention.')
    group_common.add_argument('--knn_distance_threshold', type=float, default=1.0, help='Optional: Maximum distance for a line to be considered a potential neighbor for KNN attention. Lines further than this are masked out before top-K selection.')
    group_common.add_argument('--norm_first', action='store_true', default=False, help='Use Pre-LayerNormalization instead of Post-LN (ReZero). May improve stability.')
    group_common.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    group_common.add_argument('--inference_output_dir_base_name', type=str, default="inference_output_with_probabilities", help='Base name for the directory where inference outputs with probabilities will be saved.')
    group_common.add_argument('--mlp_head_dims', type=int, nargs='+', default=[128, 64, 32], help='List of hidden layer sizes for an MLP classification head. e.g., --mlp_head_dims 512 256. If empty, a single linear layer is used.')

    # Training Arguments
    group_train = parser.add_argument_group('Training Arguments')
    group_train.add_argument('--best_model_base_name', type=str, default="best_model_orient_GraphSeq", help='Base name best model')
    group_train.add_argument('--latest_ckpt_base_name', type=str, default="latest_ckpt_orient_GraphSeq", help='Base name latest checkpoint')
    group_train.add_argument('--plot_base_name_train', type=str, default="cm_orient_GraphSeq", help='Base name train CM plot')
    group_train.add_argument('--plot_base_name_convergence', type=str, default="convergence_orient_GraphSeq", help='Base name convergence plot')
    group_train.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
    group_train.add_argument('--batch_size', type=int, default=128, help='Batch size')
    group_train.add_argument('--epochs', type=int, default=100, help='Max epochs')
    group_train.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    group_train.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio')
    group_train.add_argument('--lr_patience', type=int, default=5, help='LR scheduler patience')
    group_train.add_argument('--lr_factor', type=float, default=0.1, help='LR reduction factor')
    group_train.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    group_train.add_argument('--max_class_weight', type=float, default=None, help='Max log class weight clip value')
    group_train.add_argument('--label_smoothing', type=float, default=0.2, help='Label smoothing factor (e.g., 0.1). 0.0 means no smoothing.')

    # Online GEOMETRIC Noise Augmentation
    group_geom_noise = parser.add_argument_group('Online GEOMETRIC Noise Augmentation Parameters')
    group_geom_noise.add_argument('--apply_geom_noise', action='store_true', default=False, help='TRAINING ONLY: Enable geometric noise.') # Default False matches GeomNoiseOff
    group_geom_noise.add_argument('--geom_noise_global_p', type=float, default=0.7, help='TRAINING ONLY: Overall probability that any geometric noise is applied.')
    group_geom_noise.add_argument('--geom_noise_p_delete_edge', type=float, default=0.05, help='Prob of geometric edge DELETION.')
    group_geom_noise.add_argument('--geom_noise_delete_edge_ratio', type=float, default=0.05, help='Fraction of edges to DELETE.')
    group_geom_noise.add_argument('--geom_noise_p_break_edge', type=float, default=0.05, help='Prob of geometric edge BREAKING.')
    group_geom_noise.add_argument('--geom_noise_break_edge_ratio', type=float, default=0.05, help='Fraction of edges to BREAK.')
    group_geom_noise.add_argument('--geom_noise_break_length_factor', type=float, default=0.05, help='Length factor for broken edges.')
    group_geom_noise.add_argument('--geom_noise_p_angle_noise', type=float, default=0.05, help='Prob of geometric angle noise.')
    group_geom_noise.add_argument('--geom_noise_angle_std', type=float, default=0.05, help='Std deviation (radians) for angle noise.')
    group_geom_noise.add_argument('--geom_noise_p_length_noise', type=float, default=0.05, help='Prob of geometric length noise.')
    group_geom_noise.add_argument('--geom_noise_length_std', type=float, default=0.05, help='Std deviation (relative) for length noise.')
    group_geom_noise.add_argument('--geom_noise_p_delete_node', type=float, default=0.05, help='Prob of geometric node DELETION.')
    group_geom_noise.add_argument('--geom_noise_delete_node_ratio', type=float, default=0.1, help='Fraction of nodes to DELETE.')
    group_geom_noise.add_argument('--geom_noise_p_subdivide_edge', type=float, default=0.05, help='Prob of geometric edge SUBDIVISION.')
    group_geom_noise.add_argument('--geom_noise_subdivide_edge_ratio', type=float, default=0.05, help='Fraction of edges to SUBDIVIDE.')
    group_geom_noise.add_argument('--geom_noise_subdivide_n_segments', type=int, default=2, help='Number of segments for subdivision (min 2).')
    group_geom_noise.add_argument('--geom_noise_p_coord_noise', type=float, default=0.05, help='TRAINING ONLY: Prob of applying direct coordinate noise (Gaussian jitter).')
    group_geom_noise.add_argument('--geom_noise_coord_std', type=float, default=0.05, help='Absolute std deviation for coordinate noise (applied to x1,y1,x2,y2 independently).')

    # Evaluation Arguments
    group_eval = parser.add_argument_group('Evaluation Arguments')
    group_eval.add_argument('--eval_scaler_base_name', type=str, default="bboxnorm_orient_Train_GraphSeq", help='Base name for scaler for eval (usually same as train)')
    group_eval.add_argument('--eval_encoder_base_name', type=str, default="bboxnorm_orient_Train_GraphSeq", help='Base name for encoder for eval (usually same as train)')
    group_eval.add_argument('--eval_model_path', type=str, default=None, help='Optional: Path to specific model for eval. If None, uses best model path based on hparams.') # Default None, will be set by hparams later
    group_eval.add_argument('--eval_plot_base_name', type=str, default="cm_eval_GraphSeq", help='Base name eval CM plot')
    group_eval.add_argument('--eval_plot_path', type=str, default=None, help='Optional: Full path eval CM plot')
    group_eval.add_argument('--eval_with_noise', action='store_true', default=False, help='Evaluate the model on the test set with online geometric noise applied (using training noise parameters).')
    group_eval.add_argument('--eval_cm_data_base_name', type=str, default="cm_data_eval_GraphSeq", help='Base name for saving numerical CM data')

    # Visualization Arguments
    group_viz = parser.add_argument_group('Visualization Arguments')
    group_viz.add_argument('--viz_graph_ids', type=str, nargs='+', default=None, help='Graph IDs to visualize.')
    group_viz.add_argument('--viz_input_dir', type=str, default=None, help='Directory containing JSONs (alternative to --viz_graph_ids).')
    group_viz.add_argument('--viz_model_path', type=str, default=None, help='Optional: Path to specific model for viz. If None, uses best model path based on hparams.') # Default None
    group_viz.add_argument('--viz_scaler_base_name', type=str, default="bboxnorm_orient_Train_GraphSeq", help='Base name for scaler for viz')
    group_viz.add_argument('--viz_encoder_base_name', type=str, default="bboxnorm_orient_Train_GraphSeq", help='Base name for encoder for viz')
    group_viz.add_argument('--viz_output_plot_base_name', type=str, default="viz_orient_GraphSeq_pred", help='Base name viz plot pattern')
    group_viz.add_argument('--viz_output_plot_path', type=str, default=None, help='Optional: Full path pattern for viz plot')
    group_viz.add_argument('--viz_show_endpoints', action='store_true', default=False, help='Plot start/end point markers')
    group_viz.add_argument('--viz_force_delete_edge', action='store_true', default=False, help='VIZ ONLY: Force edge deletion noise.')
    group_viz.add_argument('--viz_force_break_edge', action='store_true', default=False, help='VIZ ONLY: Force edge breaking noise.')
    group_viz.add_argument('--viz_force_angle_noise', action='store_true', default=False, help='VIZ ONLY: Force angle noise.')
    group_viz.add_argument('--viz_force_length_noise', action='store_true', default=False, help='VIZ ONLY: Force length noise.')
    group_viz.add_argument('--viz_force_delete_node', action='store_true', default=False, help='VIZ ONLY: Force node deletion noise.')
    group_viz.add_argument('--viz_force_subdivide_edge', action='store_true', default=False, help='VIZ ONLY: Force edge subdivision noise.')
    group_viz.add_argument('--viz_force_coord_noise', action='store_true', default=False, help='VIZ ONLY: Force direct coordinate noise.')

    # Preprocessing Arguments
    group_prep = parser.add_argument_group('Preprocessing Arguments')
    group_prep.add_argument('--prep_input_dir', type=str, default="C:/Users/Alireza/Desktop/New folder (4)/VF_Dataset/Train", help='Input dir for preprocess')
    group_prep.add_argument('--prep_json_pattern', type=str, default="*_Graph_Classified.json", help='JSON pattern for preprocess')
    group_prep.add_argument('--prep_output_dir', type=str, default="dataset", help='Output dir for preprocess')
    group_prep.add_argument('--prep_norm_method', type=str, default="bbox_max_dim", choices=["bbox_max_dim", "normlen"], help='Normalization method')
    group_prep.add_argument('--prep_output_suffix', type=str, default="_orient_Train", help='Suffix for preprocess output filenames')
    return parser


# Argument Parser Setup
def setup_arg_parser(parser_args_list=None):
    """
    Sets up the argument parser, parses arguments, and constructs full paths.
    Args:
        parser_args_list (list, optional): List of strings to parse.
                                         If None, sys.argv[1:] is used.
                                         If [], defaults are parsed.
    """
    parser = get_configured_parser()
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='show this help message and exit')

    try:
        args = parser.parse_args(args=parser_args_list)
    except SystemExit as e:
         print(f"Argument parsing error: {e}")
         sys.exit(1)

    # Input Validation for Visualization
    current_argv = sys.argv[1:] if parser_args_list is None else parser_args_list
    viz_flags_present_in_cmd = any(arg.startswith('--viz_') for arg in current_argv)
    
    # Check if viz operation is intended based on parsed args (relevant if defaults enable viz actions)
    viz_operation_intended = (args.viz_graph_ids is not None or args.viz_input_dir is not None)

    if (viz_flags_present_in_cmd or viz_operation_intended) and \
       not args.viz_graph_ids and not args.viz_input_dir:
        # Only raise error if this is the main script parsing actual CLI args
        if parser_args_list is None:
            print("ERROR: For visualization (--viz_...), you must provide either --viz_graph_ids OR --viz_input_dir.")
            sys.exit(1)
        # else:
            # print("Warning: Visualization operation implied by args, but no input provided. Proceeding as this might be for default fetching.")

    # Construct Hyperparameter String
    try:
        lr_str = f"{args.lr:.0e}".replace('-', 'neg')
        do_str = str(args.dropout).replace('.', 'p')
        bs_str = str(args.batch_size)
        d_model_str = str(args.d_model)
        nhead_str = str(args.nhead)
        num_layers_str = str(args.num_layers)
        geom_noise_tag = "GeomNoiseOn" if args.apply_geom_noise else "GeomNoiseOff"
        norm_tag = "PreLN" if args.norm_first else "PostLN"

        attn_tag = "FullAttn"
        if args.k_nearest is not None and args.k_nearest > 0:
            attn_tag = f"knn{args.k_nearest}Seg"
            if args.knn_distance_threshold is not None:
                attn_tag += f"Th{str(args.knn_distance_threshold).replace('.', 'p')}"

        hparam_str = f"lr{lr_str}_bs{bs_str}_d{d_model_str}_nh{nhead_str}_nl{num_layers_str}_do{do_str}_{attn_tag}_{norm_tag}_logW_{geom_noise_tag}"

        if args.max_class_weight is not None:
             hparam_str += f"_maxW{str(args.max_class_weight).replace('.', 'p')}"
        
        # LABEL SMOOTHING TO HPARAM STRING
        if args.label_smoothing > 0.0:
            hparam_str += f"_ls{str(args.label_smoothing).replace('.', 'p')}"

        # MLP HEAD TO HPARAM STRING
        if args.mlp_head_dims:
            mlp_dims_str = '-'.join(map(str, args.mlp_head_dims))
            hparam_str += f"_mlp{mlp_dims_str}"

        if args.apply_geom_noise:
            hparam_str += f"_gP{str(args.geom_noise_global_p).replace('.', 'p')}"
            if args.geom_noise_p_delete_node > 0: hparam_str += "_gDelN"
            if args.geom_noise_p_subdivide_edge > 0: hparam_str += "_gSub"
            if args.geom_noise_p_coord_noise > 0: hparam_str += "_gCoord"
        if 'start_0_x' in FEATURE_COLS: # Check if offset features are actually used
            hparam_str += "_OffVec"
        hparam_str = re.sub(r'[<>:"/\\|?*]+', '_', hparam_str) # Sanitize
        args.hparam_str = hparam_str
    except Exception as e:
        print(f"Error generating hparam string: {e}")
        traceback.print_exc()
        args.hparam_str = "default_hparams"


    # Construct Full Paths Dynamically
    try:
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_dir)
        args.checkpoint_dir = os.path.join(args.base_dir, args.checkpoint_dir)
        args.plot_dir = os.path.join(args.base_dir, args.plot_dir)
        args.inference_output_dir_path = os.path.join(args.base_dir, args.inference_output_dir_base_name)

        # Determine base for models and plots (can be run_output_dir or default cpk/plots)
        model_output_base = args.run_output_dir if args.run_output_dir else args.checkpoint_dir
        plot_output_base = args.run_output_dir if args.run_output_dir else args.plot_dir

        model_filename_suffix = f"_{args.hparam_str}.pt"
        plot_filename_suffix = f"_{args.hparam_str}.png"

        args.best_model_path = os.path.join(model_output_base, f"{args.best_model_base_name}{model_filename_suffix}")
        args.latest_ckpt_path = os.path.join(model_output_base, f"{args.latest_ckpt_base_name}{model_filename_suffix}")

        args.plot_path_train = os.path.join(plot_output_base, f"{args.plot_base_name_train}{plot_filename_suffix}")
        args.plot_path_convergence = os.path.join(plot_output_base, f"{args.plot_base_name_convergence}{plot_filename_suffix}")

        args.train_val_data_json_path = os.path.join(args.dataset_dir, f"{args.train_val_data_json_base_name}.json")
        args.eval_data_json_path = os.path.join(args.dataset_dir, f"{args.eval_data_json_base_name}.json")
        args.viz_json_path = os.path.join(args.dataset_dir, f"{args.viz_json_base_name}.json")

        args.scaler_path = os.path.join(args.dataset_dir, f"{args.scaler_base_name}_scaler.pkl")
        args.encoder_path = os.path.join(args.dataset_dir, f"{args.encoder_base_name}_label_encoder.pkl")

        if args.resume_from and not os.path.isabs(args.resume_from):
            resume_file_name = f"{args.resume_from}{model_filename_suffix}" if not args.resume_from.endswith(".pt") else args.resume_from
            args.resume_from = os.path.join(args.checkpoint_dir, resume_file_name) # Assume resume is from general cpk

        args.eval_scaler_path = os.path.join(args.dataset_dir, f"{args.eval_scaler_base_name}_scaler.pkl")
        args.eval_encoder_path = os.path.join(args.dataset_dir, f"{args.eval_encoder_base_name}_label_encoder.pkl")

        # If --eval_model_path is not given, it defaults to None from get_configured_parser.
        # Here, we resolve it to best_model_path if it's still None.
        if args.eval_model_path is None:
            args.eval_model_path = args.best_model_path
        elif not os.path.isabs(args.eval_model_path) and not os.path.dirname(args.eval_model_path):
            # If only a filename is given for eval_model_path, assume it's in checkpoint_dir
            eval_file_name = f"{args.eval_model_path}{model_filename_suffix}" if not args.eval_model_path.endswith(".pt") else args.eval_model_path
            args.eval_model_path = os.path.join(args.checkpoint_dir, eval_file_name)
        # If it's an absolute path or relative path with a directory, it's used as is.

        # Resolve eval_plot_path
        default_eval_plot_filename = f"{args.eval_plot_base_name}{plot_filename_suffix}"
        if args.eval_plot_path is None: # Not specified at all
            args.eval_plot_path = os.path.join(plot_output_base, default_eval_plot_filename)
        elif not os.path.isabs(args.eval_plot_path) and not os.path.dirname(args.eval_plot_path): # Only filename part provided
            # If user gives "my_eval_plot.png", use that. If "my_eval_plot", append hparam suffix.
            base, ext = os.path.splitext(args.eval_plot_path)
            actual_eval_plot_filename = args.eval_plot_path if ext else f"{base}{plot_filename_suffix}"
            args.eval_plot_path = os.path.join(plot_output_base, actual_eval_plot_filename)
        # If absolute or relative with dir, use as is.

        args.eval_cm_data_path = os.path.join(plot_output_base, f"{args.eval_cm_data_base_name}_{args.hparam_str}.npy")


        args.viz_scaler_path = os.path.join(args.dataset_dir, f"{args.viz_scaler_base_name}_scaler.pkl")
        args.viz_encoder_path = os.path.join(args.dataset_dir, f"{args.viz_encoder_base_name}_label_encoder.pkl")

        if args.viz_model_path is None:
            args.viz_model_path = args.best_model_path
        elif not os.path.isabs(args.viz_model_path) and not os.path.dirname(args.viz_model_path):
            viz_file_name = f"{args.viz_model_path}{model_filename_suffix}" if not args.viz_model_path.endswith(".pt") else args.viz_model_path
            args.viz_model_path = os.path.join(args.checkpoint_dir, viz_file_name)

        default_viz_plot_filename_pattern = f"{args.viz_output_plot_base_name}_{{graph_id}}{plot_filename_suffix}"
        if args.viz_output_plot_path is None: # Not specified
            args.viz_output_plot_pattern = os.path.join(plot_output_base, default_viz_plot_filename_pattern)
        else: # User provided viz_output_plot_path
            if '{graph_id}' not in args.viz_output_plot_path: # Assume it's a directory for plots
                 args.viz_output_plot_pattern = os.path.join(args.viz_output_plot_path, default_viz_plot_filename_pattern)
            else: # Assume it's a full pattern including {graph_id}
                 args.viz_output_plot_pattern = args.viz_output_plot_path


        args.prep_output_dir = os.path.join(args.base_dir, args.prep_output_dir)

    except Exception as e:
         print(f"Error constructing file paths: {e}")
         traceback.print_exc()
         sys.exit(1)
    return args

# Helper to create directories
def ensure_dirs(args):
    try:
        dirs_to_create = []
        # Essential directories
        if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir: dirs_to_create.append(args.checkpoint_dir)
        if hasattr(args, 'plot_dir') and args.plot_dir: dirs_to_create.append(args.plot_dir)
        if hasattr(args, 'dataset_dir') and args.dataset_dir: dirs_to_create.append(args.dataset_dir)
        
        # Optional directories from args
        if hasattr(args, 'prep_output_dir') and args.prep_output_dir: dirs_to_create.append(args.prep_output_dir)
        if hasattr(args, 'run_output_dir') and args.run_output_dir: dirs_to_create.append(args.run_output_dir)
        if hasattr(args, 'inference_output_dir_path') and args.inference_output_dir_path: dirs_to_create.append(args.inference_output_dir_path)

        # Plot paths - get dirname from the fully constructed path
        for plot_path_attr in ['plot_path_train', 'plot_path_convergence', 'eval_plot_path']:
            if hasattr(args, plot_path_attr):
                path_val = getattr(args, plot_path_attr)
                if path_val: dirs_to_create.append(os.path.dirname(path_val))
        
        if hasattr(args, 'eval_with_noise') and args.eval_with_noise:
            noisy_plot_path = get_noisy_eval_plot_path(args) # This will use args.eval_plot_path
            if noisy_plot_path: dirs_to_create.append(os.path.dirname(noisy_plot_path))

        if hasattr(args, 'viz_output_plot_pattern') and args.viz_output_plot_pattern:
            # Create a dummy path to get the base directory for visualization plots
            dummy_viz_path = args.viz_output_plot_pattern.replace("{graph_id}", "example")
            dirs_to_create.append(os.path.dirname(dummy_viz_path))
            
        for d in set(dirs_to_create): # Use set to avoid redundant makedirs calls
            if d: # Ensure directory path is not None or empty
                os.makedirs(d, exist_ok=True)
                
    except OSError as e:
        print(f"Error creating directories: {e}")
    except Exception as e_gen:
        print(f"General error in ensure_dirs: {e_gen}")
        traceback.print_exc()


# Helper Function for Noisy Plot Path
def get_noisy_eval_plot_path(args):
    # Assumes args.eval_plot_path is already a full path by the time this is called
    base_path = args.eval_plot_path
    if not base_path:
        print("Warning: args.eval_plot_path is not defined when calling get_noisy_eval_plot_path.")
        # Fallback to a default construction if needed, though ideally eval_plot_path is always set
        plot_output_base = args.run_output_dir if args.run_output_dir else args.plot_dir
        plot_filename_suffix = f"_{args.hparam_str}.png" if hasattr(args, 'hparam_str') else ".png"
        eval_plot_base_name = getattr(args, 'eval_plot_base_name', 'cm_eval_GraphSeq_default')
        base_path = os.path.join(plot_output_base, f"{eval_plot_base_name}{plot_filename_suffix}")
        print(f"  Fallback noisy plot base_path: {base_path}")


    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    noisy_filename = f"{name}_noisy{ext}"
    return os.path.join(directory, noisy_filename)
