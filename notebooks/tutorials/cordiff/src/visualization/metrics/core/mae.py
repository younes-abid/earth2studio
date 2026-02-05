# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
MAE metric visualization - Mean Absolute Error.
"""

import numpy as np
from ..base_metric import BaseMetricPlot


class MAEPlot(BaseMetricPlot):
    """
    Visualizes Mean Absolute Error metric.
    """
    
    def compute_metric(self, predictions, ground_truth):
        """
        Compute MAE.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions
        ground_truth : np.ndarray
            Ground truth values
            
        Returns
        -------
        float or np.ndarray
            MAE value(s)
        """
        return np.mean(np.abs(predictions - ground_truth), axis=0)
    
    def plot(self, data, **kwargs):
        """
        Generate MAE visualization plots.
        
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
        # TODO: Implement MAE visualization
        # - Compute MAE for each ensemble member
        # - Create spatial MAE maps
        # - Show MAE distribution and summary statistics
        # - Add colorbar and title
        # - Save with descriptive filename
        pass