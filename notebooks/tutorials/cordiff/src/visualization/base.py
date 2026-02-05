# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Base classes for all visualization scripts.
Provides common functionality and ensures consistent interface.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from abc import ABC, abstractmethod
from pathlib import Path


class BasePlot(ABC):
    """
    Base class for all visualization plots.
    Provides common setup, styling, and saving functionality.
    """
    
    def __init__(self, config, output_folder):
        """
        Initialize base plot with configuration and output folder.
        
        Parameters
        ----------
        config : EnsembleConfig
            Configuration object with plotting parameters
        output_folder : str or Path
            Directory to save plots
        """
        self.config = config
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib defaults
        plt.rcParams['figure.dpi'] = config.DPI
        plt.rcParams['savefig.dpi'] = config.DPI
        plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
    
    @abstractmethod
    def plot(self, data, **kwargs):
        """
        Main plotting method - must be implemented by subclasses.
        
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
        pass
    
    def _setup_geographic_axes(self, ax, extent=None):
        """
        Set up geographic axes with cartopy if enabled.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes to configure
        extent : list, optional
            Geographic extent [lon_min, lon_max, lat_min, lat_max]
        """
        if self.config.USE_CARTOPY:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.OCEAN, alpha=0.3)
            ax.add_feature(cfeature.LAND, alpha=0.3)
            ax.gridlines(draw_labels=True, alpha=0.5)
            
            if extent:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    def _get_colormap_settings(self, data_type='variable'):
        """
        Get appropriate colormap settings based on data type.
        
        Parameters
        ----------
        data_type : str
            Type of data: 'variable', 'residual', 'std', 'metric'
            
        Returns
        -------
        dict
            Dictionary with colormap settings
        """
        cmap_map = {
            'variable': self.config.VARIABLE_CMAP,
            'residual': self.config.RESIDUAL_CMAP,
            'std': self.config.STD_CMAP,
            'metric': self.config.METRIC_CMAP
        }
        
        settings = {'cmap': cmap_map.get(data_type, 'viridis')}
        
        if self.config.USE_VMIN_VMAX and data_type == 'variable':
            settings.update({
                'vmin': self.config.VARIABLE_VMIN,
                'vmax': self.config.VARIABLE_VMAX
            })
        
        return settings
    
    def _save_plot(self, fig, filename, **save_kwargs):
        """
        Save plot with consistent naming and format.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : str
            Base filename (without extension)
        **save_kwargs
            Additional arguments for savefig
            
        Returns
        -------
        str
            Path to saved file
        """
        filepath = self.output_folder / f"{filename}.{self.config.SAVE_FORMAT}"
        
        default_kwargs = {
            'dpi': self.config.DPI,
            'bbox_inches': 'tight',
            'facecolor': 'white'
        }
        default_kwargs.update(save_kwargs)
        
        fig.savefig(filepath, **default_kwargs)
        plt.close(fig)
        
        return str(filepath)
    
    def _create_grid_subplot(self, n_items, max_cols=4):
        """
        Create optimal grid layout for multiple subplots.
        
        Parameters
        ----------
        n_items : int
            Number of items to plot
        max_cols : int
            Maximum number of columns
            
        Returns
        -------
        tuple
            (n_rows, n_cols) for subplot grid
        """
        n_cols = min(n_items, max_cols)
        n_rows = (n_items + n_cols - 1) // n_cols
        return n_rows, n_cols


class BaseMetricPlot(BasePlot):
    """
    Base class for metric visualization plots.
    Extends BasePlot with metric-specific functionality.
    """
    
    def __init__(self, config, output_folder, metric_name, metric_config=None):
        """
        Initialize metric plot.
        
        Parameters
        ----------
        config : EnsembleConfig
            Configuration object
        output_folder : str or Path
            Output directory
        metric_name : str
            Name of the metric
        metric_config : dict, optional
            Metric-specific configuration parameters
        """
        super().__init__(config, output_folder)
        self.metric_name = metric_name
        self.metric_config = metric_config or {}
    
    @abstractmethod
    def compute_metric(self, predictions, targets, **kwargs):
        """
        Compute the metric values.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions
        targets : np.ndarray
            Ground truth targets
        **kwargs
            Additional parameters
            
        Returns
        -------
        np.ndarray or dict
            Computed metric values
        """
        pass
    
    def plot_metric_timeseries(self, metric_values, times, **kwargs):
        """
        Create time series plot of metric values.
        
        Parameters
        ----------
        metric_values : np.ndarray
            Metric values over time
        times : np.ndarray
            Time points
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        str
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)
        
        ax.plot(times, metric_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{self.metric_name}')
        ax.set_title(f'{self.metric_name} Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(metric_values)
        std_val = np.std(metric_values)
        ax.axhline(mean_val, color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {mean_val:.3f}')
        ax.legend()
        
        filename = f"{self.metric_name.lower()}_timeseries"
        return self._save_plot(fig, filename)
    
    def plot_metric_spatial(self, metric_values, lat, lon, **kwargs):
        """
        Create spatial plot of metric values.
        
        Parameters
        ----------
        metric_values : np.ndarray
            2D spatial metric values
        lat, lon : np.ndarray
            Latitude and longitude coordinates
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        str
            Path to saved plot
        """
        if self.config.USE_CARTOPY:
            fig = plt.figure(figsize=self.config.FIGURE_SIZE)
            ax = plt.axes(projection=ccrs.PlateCarree())
            self._setup_geographic_axes(ax)
        else:
            fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)
        
        cmap_settings = self._get_colormap_settings('metric')
        
        im = ax.pcolormesh(lon, lat, metric_values, 
                          transform=ccrs.PlateCarree() if self.config.USE_CARTOPY else None,
                          **cmap_settings)
        
        plt.colorbar(im, ax=ax, label=self.metric_name, shrink=0.8)
        ax.set_title(f'{self.metric_name} Spatial Distribution')
        
        filename = f"{self.metric_name.lower()}_spatial"
        return self._save_plot(fig, filename)