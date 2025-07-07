# refiner_online_augmenter.py
import torch
import numpy as np
import math
import random
import refiner_config as r_config


PAD_VALUE_LABELS_AUG = -100
PAD_VALUE_FEATURES_AUG = 0.0

class RefinerOnlineFeatureAugmenter:
    def __init__(self, feature_cols_config, # This would be a list of currently selected features for the refiner
                 # Noise parameters for features
                 p_feature_gaussian_noise=0.0, gaussian_noise_std=0.01,
                 p_feature_dropout=0.0, feature_dropout_rate=0.05,
                 device='cpu'):
        self.feature_cols = feature_cols_config # The actual columns being fed to the model
        self.p_feature_gaussian_noise = p_feature_gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std
        self.p_feature_dropout = p_feature_dropout
        self.feature_dropout_rate = feature_dropout_rate
        self.device = device
        
        self.geom_feature_indices_in_current_input = []
        self.prob_feature_indices_in_current_input = []
        self.onehot_feature_indices_in_current_input = []

        for idx, col_name in enumerate(self.feature_cols):
            if col_name in r_config.GEOM_FEATURE_COLS:
                self.geom_feature_indices_in_current_input.append(idx)
            elif col_name in r_config.PROB_FEATURE_COLS:
                 self.prob_feature_indices_in_current_input.append(idx)
            elif col_name in r_config.ONEHOT_FEATURE_COLS:
                 self.onehot_feature_indices_in_current_input.append(idx)

        print("RefinerOnlineFeatureAugmenter Initialized")
        print(f"  Applying to {len(self.feature_cols)} input features.")
        print(f"  Geom feature indices: {len(self.geom_feature_indices_in_current_input)}")
        print(f"  Prob feature indices: {len(self.prob_feature_indices_in_current_input)}")
        print(f"  OneHot feature indices: {len(self.onehot_feature_indices_in_current_input)}")
        print("-------------------------------------------------")


    def apply(self, features_padded,        # [B, S, F_selected]
              gt_labels_padded,             # [B, S]
              orig_pred_idx_padded,         # [B, S]
              correctness_targets_padded,   # [B, S]
              attention_mask                # [B, S]
              ):
        
        noisy_features = features_padded.clone()
        # Other tensors are usually passed through by a feature augmenter

        batch_size, seq_len, num_selected_features = noisy_features.shape
        valid_elements_mask_expanded = (~attention_mask).unsqueeze(-1).expand_as(noisy_features) # [B,S,F_selected]

        if random.random() < self.p_feature_gaussian_noise and self.gaussian_noise_std > 0 and self.geom_feature_indices_in_current_input:
            noise = torch.randn_like(noisy_features) * self.gaussian_noise_std
            
            # Create a mask to apply noise only to geometric features and valid elements
            geom_noise_mask = torch.zeros_like(noisy_features, dtype=torch.bool)
            geom_noise_mask[:, :, self.geom_feature_indices_in_current_input] = True
            final_noise_application_mask = geom_noise_mask & valid_elements_mask_expanded
            
            noisy_features[final_noise_application_mask] += noise[final_noise_application_mask]

        if random.random() < self.p_feature_dropout and self.feature_dropout_rate > 0:
            dropout_mask_values = torch.rand_like(noisy_features) < self.feature_dropout_rate
            final_dropout_mask = dropout_mask_values & valid_elements_mask_expanded
            noisy_features[final_dropout_mask] = PAD_VALUE_FEATURES_AUG # Or 0.0

        return noisy_features, gt_labels_padded, orig_pred_idx_padded, correctness_targets_padded, attention_mask