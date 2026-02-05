# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
RMSE metric visualization - Root Mean Square Error.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base_metric import BaseMetricPlot


class RMSEPlot(BaseMetricPlot):
    """
    Visualizes Root Mean Square Error metric.
    """
    
    def compute_metric(self, predictions, ground_truth, **kwargs):
        """
        Compute RMSE between predictions and ground truth.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions (ensemble, y, x)
        ground_truth : np.ndarray
            Ground truth values (y, x)
            
        Returns
        -------
        dict
            Dictionary with RMSE values
        """
        # Calculate ensemble mean
        ensemble_mean = np.mean(predictions, axis=0)
        
        # Compute RMSE for ensemble mean
        rmse_mean = np.sqrt(np.mean((ensemble_mean - ground_truth) ** 2))
        
        # Compute spatial RMSE map (for ensemble mean)
        rmse_spatial = np.sqrt((ensemble_mean - ground_truth) ** 2)
        
        # Compute RMSE for each ensemble member
        rmse_members = np.sqrt(np.mean((predictions - ground_truth) ** 2, axis=(1, 2)))
        
        return {
            'rmse_mean': rmse_mean,
            'rmse_spatial': rmse_spatial,
            'rmse_members': rmse_members,
            'rmse_std': np.std(rmse_members),
            'ensemble_mean': ensemble_mean,
            'ground_truth': ground_truth
        }
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate RMSE visualization plots.
        
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
        print(f'🎨 Creating RMSE metric plots for time index {time_idx}...')
        
        # Try to load ground truth data
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                var_name = self.config.OUTPUT_VARIABLES[variable_idx]
                if var_name not in truth_ds.data_vars:
                    print(f'   ⚠️  Ground truth for {var_name} not available, skipping RMSE')
                    return None
                ground_truth = truth_ds[var_name].isel(time=time_idx).values
        except (OSError, KeyError):
            print(f'   ⚠️  Ground truth data not available, skipping RMSE')
            return None
        
        # Load prediction data and coordinates
        with xr.open_dataset(netcdf_path, group='prediction') as ds:
            var_name = self.config.OUTPUT_VARIABLES[variable_idx]
            predictions = ds[var_name].isel(time=time_idx).values  # Shape: (ensemble, y, x)
            
            # Get coordinates
            lat_2d, lon_2d = self._get_coordinates(ds)
        
        # Compute RMSE metrics
        metrics = self.compute_metric(predictions, ground_truth)
        
        # Create subplot layout
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': projection})
        
        # Plot 1: Spatial RMSE map
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[0])
            im1 = axes[0].pcolormesh(lon_2d, lat_2d, metrics['rmse_spatial'], 
                                    transform=ccrs.PlateCarree(), cmap='viridis')
        else:
            im1 = axes[0].imshow(metrics['rmse_spatial'], cmap='viridis', origin='lower')
        
        axes[0].set_title(f'{var_name} - Spatial RMSE Map')
        plt.colorbar(im1, ax=axes[0], shrink=0.8, label='RMSE')
        
        # Add statistics text
        axes[0].text(0.02, 0.98, f'Mean RMSE: {metrics["rmse_mean"]:.4f}', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: RMSE distribution across ensemble members
        axes[1].hist(metrics['rmse_members'], bins=min(10, len(metrics['rmse_members'])), 
                    alpha=0.7, edgecolor='black', color='skyblue')
        axes[1].axvline(metrics['rmse_mean'], color='red', linestyle='--', linewidth=2, 
                       label=f'Ensemble Mean RMSE: {metrics["rmse_mean"]:.4f}')
        axes[1].set_xlabel('RMSE Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{var_name} - RMSE Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add summary statistics
        stats_text = (f'Mean: {metrics["rmse_mean"]:.4f}\n'
                     f'Std: {metrics["rmse_std"]:.4f}\n'
                     f'Min: {np.min(metrics["rmse_members"]):.4f}\n'
                     f'Max: {np.max(metrics["rmse_members"]):.4f}')
        axes[1].text(0.98, 0.98, stats_text, transform=axes[1].transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Root Mean Square Error Analysis (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"rmse_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ RMSE plot saved: {os.path.basename(output_path)}')
        
        return output_path
    
    def _get_coordinates(self, ds):
        """Helper method to get coordinates from dataset"""
        # Get coordinates - check both coords and data_vars
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