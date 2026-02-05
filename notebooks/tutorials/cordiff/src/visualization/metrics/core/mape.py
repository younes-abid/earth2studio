# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
MAPE metric visualization - Mean Absolute Percentage Error.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base_metric import BaseMetricPlot


class MAPEPlot(BaseMetricPlot):
    """
    Visualizes MAPE (Mean Absolute Percentage Error) metric.
    """
    
    def compute_metric(self, predictions, ground_truth, epsilon=1e-8, **kwargs):
        """
        Compute MAPE between predictions and ground truth.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions (ensemble, y, x)
        ground_truth : np.ndarray
            Ground truth values (y, x)
        epsilon : float
            Small value to avoid division by zero
            
        Returns
        -------
        dict
            Dictionary with MAPE values
        """
        # Calculate ensemble mean
        ensemble_mean = np.mean(predictions, axis=0)
        
        # Avoid division by zero by adding epsilon
        ground_truth_safe = np.where(np.abs(ground_truth) < epsilon, 
                                    epsilon, ground_truth)
        
        # Compute MAPE for ensemble mean
        mape_spatial = np.abs((ensemble_mean - ground_truth) / ground_truth_safe) * 100
        mape_mean = np.mean(mape_spatial)
        
        # Compute MAPE for each ensemble member
        mape_members = []
        for i in range(predictions.shape[0]):
            member_mape = np.abs((predictions[i] - ground_truth) / ground_truth_safe) * 100
            mape_members.append(np.mean(member_mape))
        
        mape_members = np.array(mape_members)
        
        return {
            'mape_mean': mape_mean,
            'mape_spatial': mape_spatial,
            'mape_members': mape_members,
            'mape_std': np.std(mape_members),
            'ensemble_mean': ensemble_mean,
            'ground_truth': ground_truth,
            'epsilon': epsilon
        }
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate MAPE visualization plots.
        
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
        print(f'🎨 Creating MAPE metric plots for time index {time_idx}...')
        
        # Get epsilon from config or kwargs
        epsilon = kwargs.get('epsilon', self.config.PLOT_CORE_METRICS.get('MAPE', {}).get('epsilon', 1e-8))
        
        # Try to load ground truth data
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                var_name = self.config.OUTPUT_VARIABLES[variable_idx]
                if var_name not in truth_ds.data_vars:
                    print(f'   ⚠️  Ground truth for {var_name} not available, skipping MAPE')
                    return None
                ground_truth = truth_ds[var_name].isel(time=time_idx).values
        except (OSError, KeyError):
            print(f'   ⚠️  Ground truth data not available, skipping MAPE')
            return None
        
        # Load prediction data and coordinates
        with xr.open_dataset(netcdf_path, group='prediction') as ds:
            var_name = self.config.OUTPUT_VARIABLES[variable_idx]
            predictions = ds[var_name].isel(time=time_idx).values  # Shape: (ensemble, y, x)
            
            # Get coordinates
            lat_2d, lon_2d = self._get_coordinates(ds)
        
        # Compute MAPE metrics
        metrics = self.compute_metric(predictions, ground_truth, epsilon=epsilon)
        
        # Create subplot layout
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': projection})
        
        # Plot 1: Spatial MAPE map
        # Cap extreme values for better visualization
        mape_spatial_capped = np.clip(metrics['mape_spatial'], 0, np.percentile(metrics['mape_spatial'], 95))
        
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[0])
            im1 = axes[0].pcolormesh(lon_2d, lat_2d, mape_spatial_capped, 
                                    transform=ccrs.PlateCarree(), cmap='hot')
        else:
            im1 = axes[0].imshow(mape_spatial_capped, cmap='hot', origin='lower')
        
        axes[0].set_title(f'{var_name} - Spatial MAPE Map (%)')
        plt.colorbar(im1, ax=axes[0], shrink=0.8, label='MAPE (%)')
        
        # Add statistics text
        axes[0].text(0.02, 0.98, f'Mean MAPE: {metrics["mape_mean"]:.2f}%', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: MAPE distribution across ensemble members
        axes[1].hist(metrics['mape_members'], bins=min(10, len(metrics['mape_members'])), 
                    alpha=0.7, edgecolor='black', color='orange')
        axes[1].axvline(metrics['mape_mean'], color='red', linestyle='--', linewidth=2, 
                       label=f'Ensemble Mean MAPE: {metrics["mape_mean"]:.2f}%')
        axes[1].set_xlabel('MAPE Value (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{var_name} - MAPE Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add summary statistics
        stats_text = (f'Mean: {metrics["mape_mean"]:.2f}%\n'
                     f'Std: {metrics["mape_std"]:.2f}%\n'
                     f'Min: {np.min(metrics["mape_members"]):.2f}%\n'
                     f'Max: {np.max(metrics["mape_members"]):.2f}%\n'
                     f'ε: {epsilon:.0e}')
        axes[1].text(0.98, 0.98, stats_text, transform=axes[1].transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Mean Absolute Percentage Error Analysis (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"mape_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ MAPE plot saved: {os.path.basename(output_path)}')
        
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