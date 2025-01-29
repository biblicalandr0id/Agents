import torch
import numpy as np
import scipy.stats as stats

# --- NEURAL DIAGNOSTICS (neural_diagnostics.py) ---
class NeuralDiagnostics:
    def __init__(self, network, diagnostic_config=None):
        self.network = network

        self.diagnostic_config = diagnostic_config or {
            'gradient_norm_threshold': 10.0,
            'activation_sparsity_threshold': 0.3,
            'weight_divergence_threshold': 5.0,
            'anomaly_detection_sensitivity': 0.95
        }

        self.diagnostic_history = {
            'gradient_norms': [],
            'activation_sparsity': [],
            'weight_distributions': [],
            'loss_curvature': [],
            'feature_importance': []
        }

        self.anomaly_detectors = {
            'gradient_norm': self._detect_gradient_anomalies,
            'activation_sparsity': self._detect_sparsity_anomalies,
            'weight_distribution': self._detect_weight_distribution_anomalies
        }
        self.metrics_history = {} # Track full history for comprehensive report


    def monitor_network_health(self, inputs, targets, context, epoch=None): # Added epoch info
        """Enhanced network health monitoring with comprehensive metrics."""
        diagnostics = {
            'gradient_analysis': self._analyze_gradients(),
            'activation_analysis': self._analyze_activations(inputs, context),
            'weight_analysis': self._analyze_weight_distributions(),
            'loss_landscape': self._analyze_loss_landscape(inputs, targets, context), # Pass context here now!
            'anomalies': self._detect_network_anomalies()
        }

        # Update diagnostic history and metrics_history
        for key, value in diagnostics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    history_key = f"{key}_{subkey}" # Create unique history key
                    if history_key not in self.diagnostic_history:
                        self.diagnostic_history[history_key] = []
                        self.metrics_history[