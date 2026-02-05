# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
CRPS metric visualization - Continuous Ranked Probability Score.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from scipy import stats
from ..base_metric import BaseMetricPlot


class CRPSPlot(BaseMetricPlot):
    """
    Visualizes CRPS metric for ensemble forecasts.
    """
    
    def compute_metric(self, predictions, ground_truth, **kwargs):
        """
        Compute CRPS for ensemble predictions.
        
        Parameters
        ----------
        predictions : np.ndarray
            Ensemble predictions (ensemble_size, y, x)
        ground_truth : np.ndarray
            Ground truth values (y, x)
            
        Returns
        -------
        dict
            Dictionary with CRPS values and related metrics
        """
        # Sort ensemble predictions along the first axis
        sorted_preds = np.sort(predictions, axis=0)
        ensemble_size = sorted_preds.shape[0]
        
        # Compute CRPS using the integral formula
        # CRPS = E[|X - Y|] - 0.5 * E[|X - X'|]
        # where X is forecast, Y is observation, X' is independent forecast
        
        # First term: mean absolute error between each member and truth
        mae_term = np.mean(np.abs(predictions - ground_truth), axis=0)
        
        # Second term: mean absolute difference between ensemble members
        spread_term = 0.0
        for i in range(ensemble_size):
            for j in range(ensemble_size):
                spread_term += np.abs(sorted_preds[i] - sorted_preds[j])
        spread_term /= (2 * ensemble_size ** 2)
        
        # CRPS = first term - second term
        crps_spatial = mae_term - spread_term
        
        # Global CRPS (spatial average)
        crps_global = np.mean(crps_spatial)
        
        # Compute ensemble mean for comparison
        ensemble_mean = np.mean(predictions, axis=0)
        
        # Compute MAE of ensemble mean for skill comparison
        mae_ensemble_mean = np.mean(np.abs(ensemble_mean - ground_truth))
        
        # CRPS skill score relative to ensemble mean
        crps_skill = 1 - (crps_global / mae_ensemble_mean) if mae_ensemble_mean > 0 else 0
        
        return {
            'crps_spatial': crps_spatial,
            'crps_global': crps_global,
            'crps_skill': crps_skill,
            'mae_ensemble_mean': mae_ensemble_mean,
            'ensemble_mean': ensemble_mean,
            'ground_truth': ground_truth,
            'crps_std': np.std(crps_spatial),
            'crps_percentiles': np.percentile(crps_spatial, [25, 50, 75])
        }
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate CRPS visualization plots.
        
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
        print(f'🎨 Creating CRPS metric plots for time index {time_idx}...')
        
        # Try to load ground truth data
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                var_name = self.config.OUTPUT_VARIABLES[variable_idx]
                if var_name not in truth_ds.data_vars:
                    print(f'   ⚠️  Ground truth for {var_name} not available, skipping CRPS')
                    return None
                ground_truth = truth_ds[var_name].isel(time=time_idx).values
        except (OSError, KeyError):
            print(f'   ⚠️  Ground truth data not available, skipping CRPS')
            return None
        
        # Load prediction data and coordinates
        with xr.open_dataset(netcdf_path, group='prediction') as ds:
            var_name = self.config.OUTPUT_VARIABLES[variable_idx]
            predictions = ds[var_name].isel(time=time_idx).values  # Shape: (ensemble, y, x)
            
            # Get coordinates
            lat_2d, lon_2d = self._get_coordinates(ds)
        
        # Compute CRPS metrics
        metrics = self.compute_metric(predictions, ground_truth)
        
        # Create subplot layout
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': projection})
        axes = axes.flatten()
        
        # Plot 1: Spatial CRPS map
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[0])
            im1 = axes[0].pcolormesh(lon_2d, lat_2d, metrics['crps_spatial'], 
                                   transform=ccrs.PlateCarree(), cmap='viridis')
        else:
            im1 = axes[0].imshow(metrics['crps_spatial'], cmap='viridis', origin='lower')
        
        axes[0].set_title(f'{var_name} - Spatial CRPS Map')
        plt.colorbar(im1, ax=axes[0], shrink=0.8, label='CRPS')
        
        # Add statistics text
        axes[0].text(0.02, 0.98, f'Mean CRPS: {metrics["crps_global"]:.4f}', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: CRPS histogram
        crps_flat = metrics['crps_spatial'].flatten()
        crps_valid = crps_flat[~np.isnan(crps_flat)]
        
        axes[1].hist(crps_valid, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        axes[1].axvline(metrics['crps_global'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean CRPS: {metrics["crps_global"]:.4f}')
        axes[1].axvline(np.median(crps_valid), color='orange', linestyle='--', linewidth=2,
                       label=f'Median CRPS: {np.median(crps_valid):.4f}')
        axes[1].set_xlabel('CRPS Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{var_name} - CRPS Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot - CRPS vs MAE of ensemble mean
        mae_spatial = np.abs(metrics['ensemble_mean'] - ground_truth)
        
        # Sample points for cleaner visualization
        n_sample = min(1000, len(crps_valid))
        if len(crps_valid) > n_sample:
            indices = np.random.choice(len(crps_valid), n_sample, replace=False)
            crps_sample = crps_valid[indices]
            mae_sample = mae_spatial.flatten()[~np.isnan(crps_flat)][indices]
        else:
            crps_sample = crps_valid
            mae_sample = mae_spatial.flatten()[~np.isnan(crps_flat)]
        
        axes[2].scatter(mae_sample, crps_sample, alpha=0.5, s=2)
        
        # Add diagonal line (CRPS = MAE, no ensemble improvement)
        min_val = min(np.min(mae_sample), np.min(crps_sample))
        max_val = max(np.max(mae_sample), np.max(crps_sample))
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', 
                    label='CRPS = MAE (no skill)')
        
        axes[2].set_xlabel('MAE of Ensemble Mean')
        axes[2].set_ylabel('CRPS')
        axes[2].set_title(f'{var_name} - CRPS vs MAE Comparison')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: CRPS skill and summary statistics
        axes[3].axis('off')
        
        # Create summary statistics text
        stats_text = f"""CRPS Analysis Summary
        
Global CRPS: {metrics['crps_global']:.4f}
MAE (Ensemble Mean): {metrics['mae_ensemble_mean']:.4f}
CRPS Skill Score: {metrics['crps_skill']:.3f}

Spatial Statistics:
  Standard Deviation: {metrics['crps_std']:.4f}
  25th Percentile: {metrics['crps_percentiles'][0]:.4f}
  50th Percentile: {metrics['crps_percentiles'][1]:.4f}
  75th Percentile: {metrics['crps_percentiles'][2]:.4f}
  
Interpretation:
  CRPS < MAE: Ensemble adds value
  CRPS ≈ MAE: Little ensemble benefit  
  CRPS > MAE: Ensemble degrades skill"""
        
        axes[3].text(0.1, 0.9, stats_text, transform=axes[3].transAxes, 
                    verticalalignment='top', fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Continuous Ranked Probability Score (CRPS) Analysis (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"crps_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ CRPS plot saved: {os.path.basename(output_path)}')
        
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