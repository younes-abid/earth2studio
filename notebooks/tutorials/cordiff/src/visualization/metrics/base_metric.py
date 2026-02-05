# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Base class for metric visualization plots.
Extends BasePlot with metric-specific functionality.
"""

from ..base import BasePlot


class BaseMetricPlot(BasePlot):
    """
    Base class for all metric visualization plots.
    Provides common functionality for metric computation and plotting.
    """
    
    def __init__(self, config, output_folder, metric_name, metric_params):
        """
        Initialize metric plot.
        
        Parameters
        ----------
        config : EnsembleConfig
            Configuration with plot settings
        output_folder : str or Path
            Output directory for plots
        metric_name : str
            Name of the metric (e.g., 'RMSE', 'MAE')
        metric_params : dict
            Parameters for metric computation
        """
        super().__init__(config, output_folder)
        self.metric_name = metric_name
        self.metric_params = metric_params
    
    def compute_metric(self, predictions, ground_truth):
        """
        Compute the metric value.
        Should be implemented by subclasses.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions
        ground_truth : np.ndarray
            Ground truth values
            
        Returns
        -------
        float or np.ndarray
            Computed metric value(s)
        """
        raise NotImplementedError("Subclasses must implement compute_metric method")
    
    def _get_filename_prefix(self):
        """Get filename prefix including metric name"""
        return f"metric_{self.metric_name.lower()}"