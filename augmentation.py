import torch
import numpy as np
import random
import torchvision.transforms as transforms

# --- ADAPTIVE DATA AUGMENTER (neural_augmentation.py) ---
class AdaptiveDataAugmenter:
    def __init__(self,
                 noise_levels=[0.1, 0.3, 0.5, 0.7],
                 augmentation_strategies=None):
        self.noise_levels = noise_levels
        self.current_noise_index = 0

        self.augmentation_strategies = augmentation_strategies or [
            'gaussian_noise',
            'dropout',
            'mixup',
            'feature_space_augmentation',
            'horizontal_flip', # New Augmentation
            'vertical_flip',   # New Augmentation
            #'temporal_warping' # Temporal warping removed from defaults
        ]
        self.current_strategy_index = 0

        self.augmentation_history = {
            'applied_strategies': [],
            'noise_levels': [],
            'performance_impact': []
        }
        self.horizontal_flip_transform = transforms.RandomHorizontalFlip(p=0.5) # Pre-initialize transforms
        self.vertical_flip_transform = transforms.RandomVerticalFlip(p=0.5)


    def augment(self, data, context=None):
        """
        Augment data with multiple strategies, including new flips.
        """
        strategy = self.augmentation_strategies[self.current_strategy_index]
        noise_scale = self.noise_levels[self.current_noise_index]

        augmented_data = self._apply_augmentation(data, strategy, noise_scale, context)

        self.augmentation_history['applied_strategies'].append(strategy)
        self.augmentation_history['noise_levels'].append(noise_scale)

        return augmented_data

    def _apply_augmentation(self, data, strategy, noise_scale, context=None):
        """Apply augmentation strategies, including flips."""
        if strategy == 'gaussian_noise':
            return data + torch.randn_like(data) * noise_scale

        elif strategy == 'dropout':
            mask = torch.bernoulli(torch.full(data.shape, 1 - noise_scale)).bool()
            return data.masked_fill(mask, 0)

        elif strategy == 'mixup':
            batch_size = data.size(0)
            shuffle_indices = torch.randperm(batch_size)
            lam = np.random.beta(noise_scale, noise_scale)
            return lam * data + (1 - lam) * data[shuffle_indices]

        elif strategy == 'feature_space_augmentation':
            transform_matrix = torch.randn_like(data) * noise_scale * data.std()
            return data + transform_matrix

        elif strategy == 'horizontal_flip': # Horizontal Flip Augmentation
            return self.horizontal_flip_transform(data)

        elif strategy == 'vertical_flip':   # Vertical Flip Augmentation
            return self.vertical_flip_transform(data)


        elif strategy == 'temporal_warping': # Temporal Warping - kept, but not in defaults
            seq_len = data.size(1)
            warp_points = np.sort(random.sample(range(seq_len), int(seq_len*noise_scale)))
            warped_data = data.clone()
            for point in warp_points:
                offset = random.choice([-1, 1])
                if 0 <= point + offset < seq_len:
                    warped_data[:, point] = (data[:, point] + data[:, point + offset]) / 2
            return warped_data

        return data


    def adjust_augmentation(self, network_performance, diagnostics=None):
        """Adjust augmentation based on performance and diagnostics."""
        base_adjustment = self._compute_adjustment(network_performance, diagnostics)

        if base_adjustment < 0.4: # Adjusted thresholds for more frequent changes
            self.current_noise_index = min(
                self.current_noise_index + 1,
                len(self.noise_levels) - 1
            )
            if base_adjustment < 0.2: # More aggressive strategy switch at lower performance
                self.current_strategy_index = (
                    self.current_strategy_index + 1
                ) % len(self.augmentation_strategies)
        elif base_adjustment > 0.7: # Adjusted thresholds
            self.current_noise_index = max(
                self.current_noise_index - 1,
                0
            )

        self.augmentation_history['performance_impact'].append(base_adjustment)
        return self.noise_levels[self.current_noise_index]

    def _compute_adjustment(self, performance, diagnostics=None):
        """Compute adjustment score based on performance and diagnostics."""
        if diagnostics is None:
            return performance

        sparsity = diagnostics.get('activation_analysis', {}).get('activation_sparsity', 0)
        curvature = diagnostics.get('loss_landscape', {}).get('loss_curvature', 0)
        grad_norm_ratio = diagnostics.get('gradient_analysis', {}).get('grad_norm_ratio', 0) # Use grad norm ratio


        # More weight on sparsity and grad_norm_ratio in adjustment
        adjustment_score = (performance + (1 - sparsity) * 2 - curvature/10 + (1-grad_norm_ratio)) / 4.5 # Adjusted weights
        return adjustment_score


    def get_augmentation_report(self):
        """Generate augmentation report."""
        return self.augmentation_history