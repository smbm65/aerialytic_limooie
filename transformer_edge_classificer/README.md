# Geometric Edge Transformer

This repository contains a PyTorch implementation of a Transformer-based model designed for classifying edges in 2D geometric graphs. The model leverages a custom K-Nearest Neighbor (KNN) attention mechanism to focus on spatially relevant context and employs extensive online geometric data augmentation to improve robustness.

The entire pipeline, from data preparation to training, evaluation, and visualization, is driven by a unified and highly configurable command-line interface.

## ✨ Key Features

-   **Custom KNN Segment Attention**: Instead of using standard full self-attention, the model calculates the minimum geometric distance between line segments. It then builds a sparse attention mask, allowing each edge to only attend to its `k` nearest neighbors. This makes the model more efficient and focuses its attention on local geometric context.
-   **Online Geometric Augmentation**: To improve model generalization and robustness to noisy data, a powerful geometric augmentation pipeline is applied on-the-fly during training. This includes operations like:
    -   Edge Deletion & Breaking
    -   Node Deletion
    -   Edge Subdivision
    -   Coordinate, Angle, and Length Jitter
-   **End-to-End Pipeline**: The repository includes scripts for the complete machine learning lifecycle:
    1.  `preprocess.py`: Processes raw JSON data into a unified format, fits scalers/encoders.
    2.  `main.py`: Handles model training, validation, and final evaluation.
    3.  `generate_predictions_json.py`: Runs inference on new data and saves outputs with class probabilities.
    4.  `visualize.py`: Generates plots comparing model predictions to ground truth.
    5.  `visualize_knn_segments_from_dataset.py`: A utility to visualize the KNN segment distance calculations.
-   **Highly Configurable**: Nearly every aspect of the model architecture, training loop, and augmentation strategy is controllable via command-line arguments defined in `config.py`. This allows for easy experimentation and hyperparameter tuning.
-   **JIT-Compiled Custom Layers**: The custom Transformer encoder layers are JIT-scripted for potential performance improvements.

## 📁 Project Structure

```
.
├── main.py                     # Main script for training and evaluation
├── model.py                    # Defines the TransformerEdgeClassifier and custom attention layers
├── config.py                   # Centralized argument parser and project configuration
├── trainer.py                  # Contains the training loop logic
├── data_utils.py               # DataLoaders, Dataset class, and online geometric augmentation
├── utils.py                    # Helper functions (plotting, checkpoints, evaluation)
├── preprocess.py               # Script for initial data preparation and feature engineering
├── generate_predictions_json.py  # Script for running inference and saving results
├── visualize.py                # Script to visualize model predictions vs. ground truth
├── visualize_knn_segments_from_dataset.py # Utility to inspect KNN attention behavior
└── online_augmenter.py         # (Optional/Future) Vectorized feature-space augmentation
```

## ⚙️ Workflow & Usage

### 1. Dependencies

First, install the required Python packages.

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm
```

### 2. Preprocessing

The first step is to process your raw graph data. The `preprocess.py` script will:
1.  Read raw `*_Graph_Classified.json` files.
2.  Perform canonical ordering and normalization of edge coordinates.
3.  Remove duplicate edges within each graph.
4.  Calculate a rich set of geometric features for each edge.
5.  Fit a `StandardScaler` on the features and a `LabelEncoder` on the class labels across the entire dataset.
6.  Save the processed data, scaler, and encoder to the `dataset/` directory.

You should run this for both your training and testing datasets.

**Example Command:**

```bash
# Preprocess the training data
python preprocess.py --prep_input_dir "VF_Dataset/Train" --prep_output_suffix "_orient_Train"   

# Preprocess the testing data
python preprocess.py --prep_input_dir "VF_Dataset/Test" --prep_output_suffix "_orient_Test"
```

### 3. Training

Once the data is preprocessed, you can train the model using `main.py`. This script handles:
-   Loading the preprocessed data, scaler, and encoder.
-   Initializing the model, optimizer, and loss function (with class weights).
-   Running the training and validation loops.
-   Applying online geometric augmentation to the training data.
-   Saving the best model based on validation loss and creating periodic checkpoints.
-   Plotting convergence curves.
-   Running a final evaluation on the clean and (optionally) noisy test set.

All hyperparameters (e.g., `d_model`, `nhead`, `k_nearest`, `lr`) are passed as command-line arguments.

**Example Command:**

```bash
# Train the model with KNN attention (k=16) and a distance threshold
python main.py --k_nearest 16 --knn_distance_threshold 1.0
```

### 4. Evaluation

After training, the best model is evaluated on the test set. This is done automatically at the end of the `main.py` script. You can also run a separate evaluation-only script if needed. Evaluation produces a classification report and a confusion matrix plot.

The `--eval_with_noise` flag allows you to assess the model's robustness by applying the same geometric augmentations to the test set that were used during training.

**Example Command (assuming a standalone `evaluate.py` or similar setup):**

```bash
# Evaluate on the clean test set
python .\evaluate.py

# Evaluate on the geometrically noisy test set
python .\evaluate.py --eval_with_noise
```

### 5. Inference

To generate predictions for new, unseen data, use the `generate_predictions_json.py` script. It loads a trained model and generates new JSON files containing the original edge geometry along with a list of class probabilities for each edge.

**Example Command:**

```bash
# Generate predictions on the test set, including a pass with geometric noise
python generate_predictions_json.py --eval_data_json_base_name visualization_data_bboxnorm_orient_Test_GraphSeq --eval_scaler_base_name bboxnorm_orient_Train_GraphSeq --eval_encoder_base_name bboxnorm_orient_Train_GraphSeq --k_nearest 16 --apply_geom_noise --geom_noise_global_p 0.7 --geom_noise_p_coord_noise 0.1 --geom_noise_coord_std 0.02
```

### 6. Visualization & Analysis

The repository includes powerful tools for visualizing results and model behavior.

#### A. Prediction Visualization (`visualize.py`)

This script creates a side-by-side plot comparing the model's predictions on a (potentially noisy) graph geometry with the original ground truth. This is invaluable for qualitative analysis. You can force specific types of geometric noise to see how they affect model output.

**Example Command:**

```bash
# Visualize predictions for several graphs, forcing multiple types of noise
python .\visualize.py --viz_show_endpoints --viz_force_delete_edge --viz_force_break_edge --viz_force_angle_noise --viz_force_length_noise --viz_graph_ids 0 2 3 4 5 6 7 8 9 10 995 996 997 998 999
```

#### B. KNN Attention Visualization (`visualize_knn_segments_from_dataset.py`)

This utility script helps you understand and debug the custom attention mechanism. It takes a specific line from a specific graph and plots it, highlighting its `K` nearest neighbors based on the segment-to-segment distance metric.

**Example Command:**

```bash
# For graph_id 0, visualize the 8 nearest neighbors of the line at index 0
python visualize_knn_segments_from_dataset.py --viz_json_path ./dataset/visualization_data_bboxnorm_orient_Test_GraphSeq.json --k_to_show 8 --output_dir ./knn_plots --show_all_indices --viz_graph_ids 0 --target_line_index_in_graph 0
```

### Configuration (`config.py`)

The `config.py` script is central to the project's flexibility. It uses `argparse` to define all configurable parameters. A key function, `setup_arg_parser()`, not only parses these arguments but also constructs a unique hyperparameter string (`hparam_str`) based on the chosen settings.

Example `hparam_str`: `lr1eneg04_bs128_d256_nh8_nl6_do0p1_knn16SegTh1p0_PostLN_logW_GeomNoiseOn...`

This string is automatically appended to saved model files, checkpoints, and plot filenames, ensuring that every experiment's artifacts are uniquely named and easily traceable to their configuration.