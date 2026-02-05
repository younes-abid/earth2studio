# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
CRPS metric visualization - Continuous Ranked Probability Score.
"""

import numpy as np
from ..base_metric import BaseMetricPlot


class CRPSPlot(BaseMetricPlot):
    """
    Visualizes CRPS metric for ensemble forecasts.
    """
    
    def compute_metric(self, predictions, ground_truth):
        """
        Compute CRPS for ensemble predictions.
        
        Parameters
        ----------
        predictions : np.ndarray
            Ensemble predictions (ensemble_size, ...)
        ground_truth : np.ndarray
            Ground truth values
            
        Returns
        -------
        float or np.ndarray
            CRPS value(s)
        """
        # Sort ensemble predictions
        sorted_preds = np.sort(predictions, axis=0)
        ensemble_size = sorted_preds.shape[0]
        
        # Compute CRPS using the integral formula
        crps = 0.0
        for i in range(ensemble_size):
            for j in range(ensemble_size):
                crps += np.abs(sorted_preds[i] - sorted_preds[j])
        crps /= (2 * ensemble_size ** 2)
        
        # Subtract the reliability term
        for i in range(ensemble_size):
            crps -= np.abs(sorted_preds[i] - ground_truth) / ensemble_size
            
        return crps
    
    def plot(self, data, **kwargs):
        """
        Generate CRPS visualization plots.
        
        Parameters
        ----------
        data : dict
            Data dictionary from inference results
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        str
            Path to saved plot file
        """
        # TODO: Implement CRPS visualization
        # - Compute CRPS for each spatial location
        # - Create spatial CRPS maps
        # - Show CRPS distribution and summary statistics
        # - Compare with reference forecast if available
        # - Add colorbar and title
        # - Save with descriptive filename
        pass