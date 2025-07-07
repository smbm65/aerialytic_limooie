# Graph Edge Classification Refiner Model

This project implements a Transformer-based model designed to **refine** the predictions of a primary graph edge classification model. It operates as a post-processing step, taking the initial model's predictions (probabilities and/or one-hot labels) along with geometric features of the graph edges as input.

The primary goal of the Refiner is to identify and correct misclassified edges from the original model, thereby improving overall classification accuracy. It uses a dual-head architecture to simultaneously learn which predictions to trust and how to re-classify the untrustworthy ones.

## Key Features

-   **Dual-Head Architecture**:
    1.  **Main Classification Head**: A multi-layer perceptron (MLP) that outputs new class predictions for edges. It is primarily trained on edges that the original model predicted incorrectly.
    2.  **Correctness Head**: A binary classification MLP that predicts whether the original model's prediction for an edge was correct or incorrect.
-   **Transformer Encoder Core**: Uses a standard Transformer Encoder architecture to process sequences of graph edges, allowing the model to learn contextual relationships between them.
-   **KNN-based Attention Mask**: An optional mechanism to constrain the self-attention mechanism, forcing each edge to only attend to its `k` nearest geometric neighbors. This focuses the model on local context and improves efficiency.
-   **Rich Feature Engineering**: Leverages a comprehensive set of geometric features calculated for each edge, including position, orientation, and relative vector information.
-   **Online Data Augmentation**: Includes on-the-fly geometric noise (coordinate perturbation) and label corruption during training to improve model robustness.
-   **Modular & Configurable**: The entire pipeline—from preprocessing to training, evaluation, and visualization—is broken into clear scripts, with extensive command-line arguments for controlling hyperparameters and behavior.

## Project Structure

```
.
├── refiner_main.py                # Main script for training and evaluating the refiner model.
├── refiner_preprocess.py          # Prepares the refiner dataset from an original model's JSON outputs.
├── refiner_evaluate.py            # Standalone script for evaluating a pre-trained refiner model.
├── refiner_visualize.py           # Script to generate comparison plots: GT vs. Input vs. Refiner Output.
│
├── refiner_model.py               # Defines the RefinerTransformerEdgeClassifier PyTorch model architecture.
├── refiner_config.py              # Centralized configuration, argument parsing, and path management.
├── refiner_data_utils.py          # PyTorch Dataset/DataLoader, collate function, and data loading utilities.
├── refiner_trainer.py             # Contains the training loop logic (train_one_epoch, train_refiner).
├── refiner_utils.py               # Helper functions for evaluation, plotting, and checkpointing.
└── refiner_online_augmenter.py    # (Scaffold) For potential future online feature augmentation.
```

## Workflow and Usage

The workflow is a linear, four-step process: **Preprocess -> Train -> Evaluate -> Visualize**.

### Step 1: Preprocess Data

First, you must process the JSON outputs from your *original* classification model to create a dataset suitable for the refiner. This script reads the original model's outputs (containing coordinates, ground truth labels, and class probabilities), calculates geometric features, and saves the data in a structured format.

```bash
python refiner_preprocess.py \
    --prep_input_dir_orig_model_output="prediction_json_outputs/clean_output/train" \
    --prep_input_dir_orig_model_output_test="prediction_json_outputs/clean_output/test" \
    --prep_output_dir="refiner_dataset" \
    --num_classes_orig_model=6
```

This will create a `refiner_dataset/` directory containing:
-   A `graph_data/` subdirectory with individual processed graph JSONs.
-   Manifest files (`..._manifest.json`) listing the graph IDs for training and testing.
-   A scaler (`..._scaler.pkl`) and label encoder (`..._label_encoder.pkl`).

### Step 2: Train the Refiner Model

Once the data is preprocessed, you can train the refiner model using `refiner_main.py`. This script handles training, validation, learning rate scheduling, and saving the best model.

```bash
python refiner_main.py \
    --dataset_dir="refiner_dataset" \
    --epochs=100 \
    --batch_size=128 \
    --lr=1e-4 \
    --d_model=256 \
    --nhead=8 \
    --num_layers=6 \
    --k_nearest=16 \
    --refiner_input_feature_mode="probs_and_onehot" \
    --main_loss_weight=2.0 \
    --correctness_loss_weight=1.0 \
    --apply_geom_noise \
    --label_corruption_frac_graphs=0.75 \
    --label_corruption_frac_edges=0.1
```
Checkpoints and plots for this run will be saved to a subdirectory inside `refiner_cpk/` named after the hyperparameter string (e.g., `refiner_cpk/Ref_lr1eneg04_bs128_...`).

### Step 3: Evaluate the Model

After training, you can run a standalone evaluation on the test set. This will print a detailed classification report and generate confusion matrices for both the main and correctness heads.

```bash
python refiner_evaluate.py \
    --dataset_dir="refiner_dataset" \
    --eval_model_path="refiner_cpk/Ref_lr1eneg04_bs128_do0p1_PaO_GeomNoiseOff_LblCorrOn_knn16.../refiner_best_model.pt" \
    --eval_with_geom_noise
```
-   Replace the `--eval_model_path` with the actual path to your trained model.
-   Use `--eval_with_geom_noise` to test the model's robustness to noisy coordinates.

### Step 4: Visualize Predictions

To qualitatively analyze the model's performance, you can generate a triple-panel plot for specific graphs, comparing the ground truth, the input to the refiner (original model's predictions), and the final refined output.

The following example visualizes `graph_id=0` from the test set, using a specific pre-trained model. It also demonstrates how to visualize the effect of applying label corruption on the fly.

```bash
python refiner_visualize.py --prep_output_dir refiner_dataset --run_output_dir refiner_plots --viz_model_path refiner_cpk/Ref_lr1eneg04_bs128_do0p1_PaO_GeomNoiseOff_LblCorrOn_knn4_kthr1p0_d128nh8nl4/refiner_best_model.pt --viz_json_base_name refiner_geom_Test_RefinerData --viz_corrupt_labels --label_corruption_frac_graphs 1.0 --label_corruption_frac_edges 0.2 --viz_graph_ids 0
```
This command will produce a plot like `refiner_plots/refiner_viz_pred_0_lblCorr.png`.

## Configuration Details

The model's behavior is controlled via command-line arguments defined in `refiner_config.py`. Here are some of the most important ones:

| Argument                            | Description                                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Model Architecture**              |                                                                                                         |
| `--d_model`                         | The embedding dimension of the Transformer model.                                                       |
| `--nhead`                           | The number of attention heads in the Transformer.                                                       |
| `--num_layers`                      | The number of Transformer encoder layers.                                                               |
| `--refiner_main_head_dims`          | Hidden layer sizes for the main classification MLP head (e.g., `128 32`).                               |
| `--refiner_correctness_head_dims`   | Hidden layer sizes for the correctness prediction MLP head (e.g., `32`).                                |
| **Input & Attention**               |                                                                                                         |
| `--refiner_input_feature_mode`      | Which features to use from the original model: `onehot_only`, `probs_only`, `probs_and_onehot`, `geom_only`. |
| `--k_nearest`                       | The number of nearest neighbors (`k`) for the KNN attention mask. Set to 0 to disable.                |
| `--knn_distance_threshold`          | Maximum distance for an edge to be considered a neighbor in the KNN mask.                               |
| **Training & Loss**                 |                                                                                                         |
| `--lr`                              | The learning rate for the Adam optimizer.                                                               |
| `--batch_size`                      | The number of graphs per batch.                                                                         |
| `--epochs`                          | The total number of training epochs.                                                                    |
| `--main_loss_weight`                | The weight factor for the main classification loss component.                                           |
| `--correctness_loss_weight`         | The weight factor for the correctness head loss component.                                              |
| **Data Augmentation**               |                                                                                                         |
| `--apply_geom_noise`                | If set, applies random Gaussian noise to coordinates during training.                                   |
| `--label_corruption_frac_graphs`    | The fraction of graphs in a batch to which label corruption is applied.                                 |
| `--label_corruption_frac_edges`     | For a selected graph, the fraction of edges whose original predictions are randomly changed.            |