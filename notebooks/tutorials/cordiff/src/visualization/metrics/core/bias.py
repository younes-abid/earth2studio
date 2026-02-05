# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
BIAS metric visualization - Mean Bias.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base_metric import BaseMetricPlot


class BIASPlot(BaseMetricPlot):
    """
    Visualizes BIAS (Mean Bias) metric.
    """
    
    def compute_metric(self, predictions, ground_truth, **kwargs):
        """
        Compute BIAS between predictions and ground truth.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions (ensemble, y, x)
        ground_truth : np.ndarray
            Ground truth values (y, x)
            
        Returns
        -------
        dict
            Dictionary with BIAS values
        """
        # Calculate ensemble mean
        ensemble_mean = np.mean(predictions, axis=0)
        
        # Compute BIAS for ensemble mean
        bias_mean = np.mean(ensemble_mean - ground_truth)
        
        # Compute spatial BIAS map (for ensemble mean)
        bias_spatial = ensemble_mean - ground_truth
        
        # Compute BIAS for each ensemble member
        bias_members = np.mean(predictions - ground_truth, axis=(1, 2))
        
        return {
            'bias_mean': bias_mean,
            'bias_spatial': bias_spatial,
            'bias_members': bias_members,
            'bias_std': np.std(bias_members),
            'ensemble_mean': ensemble_mean,
            'ground_truth': ground_truth
        }
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate BIAS visualization plots.
        
        Parameters
        ----------
        netcdf_path : str
            Path to NetCDF ensemble file
        time_idx : int
            Time index to plot
        variable_idx : int
            Variable index to plot
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        str or None
            Path to saved plot file, or None if ground truth not available
        """
        print(f'🎨 Creating BIAS metric plots for time index {time_idx}...')
        
        # Try to load ground truth data
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                var_name = self.config.OUTPUT_VARIABLES[variable_idx]
                if var_name not in truth_ds.data_vars:
                    print(f'   ⚠️  Ground truth for {var_name} not available, skipping BIAS')
                    return None
                ground_truth = truth_ds[var_name].isel(time=time_idx).values
        except (OSError, KeyError):
            print(f'   ⚠️  Ground truth data not available, skipping BIAS')
            return None
        
        # Load prediction data and coordinates
        with xr.open_dataset(netcdf_path, group='prediction') as ds:
            var_name = self.config.OUTPUT_VARIABLES[variable_idx]
            predictions = ds[var_name].isel(time=time_idx).values  # Shape: (ensemble, y, x)
            
            # Get coordinates
            lat_2d, lon_2d = self._get_coordinates(ds)
        
        # Compute BIAS metrics
        metrics = self.compute_metric(predictions, ground_truth)
        
        # Create subplot layout
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': projection})
        
        # Plot 1: Spatial BIAS map
        # Use symmetric color limits for bias (diverging colormap)
        vmax = np.nanpercentile(np.abs(metrics['bias_spatial']), 95)
        
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[0])
            im1 = axes[0].pcolormesh(lon_2d, lat_2d, metrics['bias_spatial'], 
                                    transform=ccrs.PlateCarree(), cmap='RdBu_r',
                                    vmin=-vmax, vmax=vmax)
        else:
            im1 = axes[0].imshow(metrics['bias_spatial'], cmap='RdBu_r', 
                                origin='lower', vmin=-vmax, vmax=vmax)
        
        axes[0].set_title(f'{var_name} - Spatial BIAS Map')
        plt.colorbar(im1, ax=axes[0], shrink=0.8, label='BIAS')
        
        # Add statistics text
        axes[0].text(0.02, 0.98, f'Mean BIAS: {metrics["bias_mean"]:.4f}', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: BIAS distribution across ensemble members
        axes[1].hist(metrics['bias_members'], bins=min(10, len(metrics['bias_members'])), 
                    alpha=0.7, edgecolor='black', color='lightyellow')
        axes[1].axvline(metrics['bias_mean'], color='red', linestyle='--', linewidth=2, 
                       label=f'Ensemble Mean BIAS: {metrics["bias_mean"]:.4f}')
        axes[1].axvline(0, color='black', linestyle='-', linewidth=1, 
                       label='Zero Bias')
        axes[1].set_xlabel('BIAS Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{var_name} - BIAS Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add summary statistics
        stats_text = (f'Mean: {metrics["bias_mean"]:.4f}\n'
                     f'Std: {metrics["bias_std"]:.4f}\n'
                     f'Min: {np.min(metrics["bias_members"]):.4f}\n'
                     f'Max: {np.max(metrics["bias_members"]):.4f}')
        axes[1].text(0.98, 0.98, stats_text, transform=axes[1].transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Mean Bias Analysis (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"bias_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ BIAS plot saved: {os.path.basename(output_path)}')
        
        return output_path
    
    def _get_coordinates(self, ds):
        """Helper method to get coordinates from dataset"""
        # ...existing coordinate loading logic...
        if 'lat' in ds.coords and 'lon' in ds.coords:
            if ds.coords['lat'].ndim == 2:
                lat_2d = ds.coords['lat'].values
                lon_2d = ds.coords['lon'].values
            else:
                lat_1d = ds.coords['lat'].values
                lon_1d = ds.coords['lon'].values
                lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        elif 'lat' in ds.data_vars and 'lon' in ds.data_vars:
            lat_data = ds['lat'].values
            lon_data = ds['lon'].values
            if lat_data.ndim == 2:
                lat_2d = lat_data
                lon_2d = lon_data
            else:
                lon_2d, lat_2d = np.meshgrid(lon_data, lat_data)
        else:
            lat_1d = np.linspace(*self.config.OUTPUT_GRID['lat'])
            lon_1d = np.linspace(*self.config.OUTPUT_GRID['lon'])
            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        
        return lat_2d, lon_2d