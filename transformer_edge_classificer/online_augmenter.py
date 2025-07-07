# online_augmenter.py
import torch
import numpy as np
import math
import random

import config

# Must match data_utils and trainer
PAD_VALUE_LABELS = -100
PAD_VALUE_FEATURES = 0.0

class OnlineNoiseAugmenter:
    """
    Applies noise augmentation online to batches of graph edge sequences during training.
    Operates directly on feature tensors using highly efficient vectorized operations.
    The current implementation is already optimized using PyTorch's best practices.
    """
    def __init__(self, feature_cols,
                 p_delete_edge=0.0,
                 delete_edge_ratio=0.05,
                 p_break_edge=0.0,
                 break_edge_ratio=0.05,
                 break_length_factor=0.1,
                 p_angle_noise=0.0,
                 angle_noise_std=0.1,
                 p_length_noise=0.0,
                 length_noise_std=0.1,
                 device='cpu'):
        self.feature_cols = feature_cols
        self.p_delete_edge = p_delete_edge
        self.delete_edge_ratio = delete_edge_ratio
        self.p_break_edge = p_break_edge
        self.break_edge_ratio = break_edge_ratio
        self.break_length_factor = break_length_factor
        self.p_angle_noise = p_angle_noise
        self.angle_noise_std = angle_noise_std
        self.p_length_noise = p_length_noise
        self.length_noise_std = length_noise_std
        self.device = device

        self._get_feature_indices()
        self.angle_related_indices = self._get_angle_related_indices()
        print("OnlineNoiseAugmenter Initialized (Vectorized)")
        print(f"  Device: {self.device}")
        print(f"  Angle Related Indices: {self.angle_related_indices}")
        print("------------------------------------------------------")


    def _get_feature_indices(self):
        """Finds the column indices for features needed for noise application."""
        self.feature_indices = {}
        required_features = [
            'length', 'angle_rad', 'sin_angle', 'cos_angle',
            'abs_sin_angle', 'abs_cos_angle', 'is_horizontal',
            'is_vertical', 'is_positive_slope', 'is_negative_slope'
        ]
        for feat in required_features:
            try:
                self.feature_indices[feat] = self.feature_cols.index(feat)
            except ValueError:
                print(f"Warning: Feature '{feat}' not found in feature_cols. Some noise types may not work correctly.")
                self.feature_indices[feat] = -1

    def _get_angle_related_indices(self):
        """Gets a list of valid indices for angle-related features."""
        angle_indices = [
            idx for name, idx in self.feature_indices.items()
            if ('angle' in name or 'slope' in name or 'horizontal' in name or 'vertical' in name)
            and idx != -1
        ]
        return angle_indices

    def _recalculate_angle_features(self, features):
        """
        Recalculates derived angle features based on the (potentially noisy) 'angle_rad'.
        Operates in place on the features tensor.
        """
        angle_rad_idx = self.feature_indices.get('angle_rad', -1)
        if angle_rad_idx == -1:
            print("Warning: 'angle_rad' feature index missing, cannot recalculate angle features.")
            return

        noisy_angle_rad = features[..., angle_rad_idx]
        new_cos_angle = torch.cos(noisy_angle_rad)
        new_sin_angle = torch.sin(noisy_angle_rad)

        if self.feature_indices.get('cos_angle', -1) != -1: features[..., self.feature_indices['cos_angle']] = new_cos_angle
        if self.feature_indices.get('sin_angle', -1) != -1: features[..., self.feature_indices['sin_angle']] = new_sin_angle
        if self.feature_indices.get('abs_cos_angle', -1) != -1: features[..., self.feature_indices['abs_cos_angle']] = torch.abs(new_cos_angle)
        if self.feature_indices.get('abs_sin_angle', -1) != -1: features[..., self.feature_indices['abs_sin_angle']] = torch.abs(new_sin_angle)

        is_horizontal_flag = torch.abs(new_sin_angle) < config.ORIENT_TOLERANCE
        is_vertical_flag = torch.abs(new_cos_angle) < config.ORIENT_TOLERANCE

        if self.feature_indices.get('is_horizontal', -1) != -1: features[..., self.feature_indices['is_horizontal']] = is_horizontal_flag.float()
        if self.feature_indices.get('is_vertical', -1) != -1: features[..., self.feature_indices['is_vertical']] = is_vertical_flag.float()

        not_axial = ~is_horizontal_flag & ~is_vertical_flag
        dx_sign = torch.sign(new_cos_angle + 1e-9 * (new_cos_angle == 0))
        dy_sign = torch.sign(new_sin_angle + 1e-9 * (new_sin_angle == 0))
        sign_product = dx_sign * dy_sign

        if self.feature_indices.get('is_positive_slope', -1) != -1:
            is_positive = (sign_product > 0) & not_axial
            features[..., self.feature_indices['is_positive_slope']] = is_positive.float()
        if self.feature_indices.get('is_negative_slope', -1) != -1:
            is_negative = (sign_product < 0) & not_axial
            features[..., self.feature_indices['is_negative_slope']] = is_negative.float()


    def apply(self, features, labels, attention_mask):
        """
        Applies configured noise types to the batch based on probabilities.
        """
        noisy_features = features.clone()
        noisy_labels = labels.clone()
        noisy_mask = attention_mask.clone()

        batch_size, seq_len, feat_dim = noisy_features.shape

        if random.random() < self.p_delete_edge and self.delete_edge_ratio > 0:
            valid_mask = ~noisy_mask
            delete_prob_mask = torch.rand(batch_size, seq_len, device=self.device) < self.delete_edge_ratio
            delete_selection_mask = delete_prob_mask & valid_mask

            if delete_selection_mask.any():
                noisy_features[delete_selection_mask] = PAD_VALUE_FEATURES
                noisy_labels[delete_selection_mask] = PAD_VALUE_LABELS
                noisy_mask = noisy_mask | delete_selection_mask

        len_idx = self.feature_indices.get('length', -1)
        if random.random() < self.p_break_edge and self.break_edge_ratio > 0 and len_idx != -1:
            eligible_mask = ~noisy_mask
            break_prob_mask = torch.rand(batch_size, seq_len, device=self.device) < self.break_edge_ratio
            break_selection_mask = break_prob_mask & eligible_mask

            if break_selection_mask.any():
                noisy_features[break_selection_mask, len_idx] *= self.break_length_factor
                for angle_idx in self.angle_related_indices:
                     noisy_features[break_selection_mask, angle_idx] = 0.0

                horiz_idx = self.feature_indices.get('is_horizontal', -1)
                cos_idx = self.feature_indices.get('cos_angle', -1)
                abs_cos_idx = self.feature_indices.get('abs_cos_angle', -1)

                if horiz_idx != -1: noisy_features[break_selection_mask, horiz_idx] = 1.0
                if cos_idx != -1: noisy_features[break_selection_mask, cos_idx] = 1.0
                if abs_cos_idx != -1: noisy_features[break_selection_mask, abs_cos_idx] = 1.0

        angle_rad_idx = self.feature_indices.get('angle_rad', -1)
        if random.random() < self.p_angle_noise and self.angle_noise_std > 0 and angle_rad_idx != -1:
            valid_elements_mask = ~noisy_mask
            noise = torch.randn(batch_size, seq_len, device=self.device) * self.angle_noise_std * valid_elements_mask.float()
            noisy_features[..., angle_rad_idx] += noise
            angles_to_wrap = noisy_features[..., angle_rad_idx][valid_elements_mask]
            wrapped_angles = torch.remainder(angles_to_wrap + math.pi, 2 * math.pi) - math.pi
            noisy_features[..., angle_rad_idx][valid_elements_mask] = wrapped_angles
            self._recalculate_angle_features(noisy_features)

        if random.random() < self.p_length_noise and self.length_noise_std > 0 and len_idx != -1:
            valid_elements_mask = ~noisy_mask
            original_lengths = noisy_features[..., len_idx] * valid_elements_mask.float()
            noise_std_dev = original_lengths * self.length_noise_std
            noise = torch.randn(batch_size, seq_len, device=self.device) * noise_std_dev
            noisy_features[..., len_idx] += noise
            noisy_features[..., len_idx] = torch.clamp(noisy_features[..., len_idx], min=0.0)

            zero_length_mask = (noisy_features[..., len_idx] < config.COORD_TOLERANCE) & valid_elements_mask
            if zero_length_mask.any():
                for angle_idx in self.angle_related_indices:
                     noisy_features[zero_length_mask, angle_idx] = 0.0
                horiz_idx = self.feature_indices.get('is_horizontal', -1)
                cos_idx = self.feature_indices.get('cos_angle', -1)
                abs_cos_idx = self.feature_indices.get('abs_cos_angle', -1)
                if horiz_idx != -1: noisy_features[zero_length_mask, horiz_idx] = 1.0
                if cos_idx != -1: noisy_features[zero_length_mask, cos_idx] = 1.0
                if abs_cos_idx != -1: noisy_features[zero_length_mask, abs_cos_idx] = 1.0
                if angle_rad_idx != -1: noisy_features[zero_length_mask, angle_rad_idx] = 0.0


        return noisy_features, noisy_labels, noisy_mask
