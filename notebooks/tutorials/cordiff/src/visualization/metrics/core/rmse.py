# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
RMSE metric visualization - Root Mean Square Error.
"""

import numpy as np
from ..base_metric import BaseMetricPlot


class RMSEPlot(BaseMetricPlot):
    """
    Visualizes Root Mean Square Error metric.
    """
    
    def compute_metric(self, predictions, ground_truth):
        """
        Compute RMSE.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions
        ground_truth : np.ndarray
            Ground truth values
            
        Returns
        -------
        float or np.ndarray
            RMSE value(s)
        """
        return np.sqrt(np.mean((predictions - ground_truth) ** 2, axis=0))
    
    def plot(self, data, **kwargs):
        """
        Generate RMSE visualization plots.
        
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
        # TODO: Implement RMSE visualization
        # - Compute RMSE for each ensemble member
        # - Create spatial RMSE maps
        # - Show RMSE distribution and summary statistics
        # - Add colorbar and title
        # - Save with descriptive filename
        pass