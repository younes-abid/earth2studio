# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Spread metric visualization - Ensemble spread analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base_metric import BaseMetricPlot


class SpreadPlot(BaseMetricPlot):
    """
    Visualizes ensemble spread metrics.
    """
    
    def compute_metric(self, predictions, ground_truth=None, **kwargs):
        """
        Compute ensemble spread.
        
        Parameters
        ----------
        predictions : np.ndarray
            Ensemble predictions (ensemble_size, y, x)
        ground_truth : np.ndarray, optional
            Ground truth values (not used for spread but useful for spread-skill analysis)
            
        Returns
        -------
        dict
            Dictionary with ensemble spread metrics
        """
        # Compute standard deviation across ensemble members (spread)
        spread = np.std(predictions, axis=0)
        
        # Compute variance (for variance-based metrics)
        variance = np.var(predictions, axis=0)
        
        # Compute ensemble mean
        ensemble_mean = np.mean(predictions, axis=0)
        
        # Compute range (max - min)
        ensemble_range = np.max(predictions, axis=0) - np.min(predictions, axis=0)
        
        # Compute interquartile range
        q75 = np.percentile(predictions, 75, axis=0)
        q25 = np.percentile(predictions, 25, axis=0)
        iqr = q75 - q25
        
        # Global statistics
        mean_spread = np.mean(spread)
        mean_variance = np.mean(variance)
        mean_range = np.mean(ensemble_range)
        mean_iqr = np.mean(iqr)
        
        results = {
            'spread': spread,
            'variance': variance,
            'ensemble_mean': ensemble_mean,
            'ensemble_range': ensemble_range,
            'iqr': iqr,
            'q25': q25,
            'q75': q75,
            'mean_spread': mean_spread,
            'mean_variance': mean_variance,
            'mean_range': mean_range,
            'mean_iqr': mean_iqr,
            'spread_percentiles': np.percentile(spread, [25, 50, 75]),
            'coefficient_of_variation': spread / (np.abs(ensemble_mean) + 1e-8)
        }
        
        # If ground truth is provided, compute spread-skill relationship
        if ground_truth is not None:
            # Compute error of ensemble mean
            error = np.abs(ensemble_mean - ground_truth)
            
            # Correlation between spread and error
            spread_flat = spread.flatten()
            error_flat = error.flatten()
            
            # Remove NaN values
            valid_mask = ~(np.isnan(spread_flat) | np.isnan(error_flat))
            if np.sum(valid_mask) > 0:
                spread_skill_corr = np.corrcoef(spread_flat[valid_mask], error_flat[valid_mask])[0, 1]
                
                # Spread-skill ratio (should be close to 1 for well-calibrated ensembles)
                spread_skill_ratio = np.mean(spread) / np.mean(error) if np.mean(error) > 0 else np.inf
            else:
                spread_skill_corr = np.nan
                spread_skill_ratio = np.nan
            
            results.update({
                'error': error,
                'spread_skill_corr': spread_skill_corr,
                'spread_skill_ratio': spread_skill_ratio,
                'mean_error': np.mean(error)
            })
        
        return results
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate ensemble spread visualization plots.
        
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
        str
            Path to saved plot file
        """
        print(f'🎨 Creating ensemble spread plots for time index {time_idx}...')
        
        # Load prediction data and coordinates
        with xr.open_dataset(netcdf_path, group='prediction') as ds:
            var_name = self.config.OUTPUT_VARIABLES[variable_idx]
            predictions = ds[var_name].isel(time=time_idx).values  # Shape: (ensemble, y, x)
            
            # Get coordinates
            lat_2d, lon_2d = self._get_coordinates(ds)
        
        # Try to load ground truth data for spread-skill analysis
        ground_truth = None
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                if var_name in truth_ds.data_vars:
                    ground_truth = truth_ds[var_name].isel(time=time_idx).values
                    print(f'   📊 Ground truth available, including spread-skill analysis')
        except (OSError, KeyError):
            print(f'   📊 Ground truth not available, showing spread analysis only')
        
        # Compute spread metrics
        metrics = self.compute_metric(predictions, ground_truth)
        
        # Create subplot layout
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        n_plots = 4 if ground_truth is not None else 3
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': projection})
        axes = axes.flatten()
        
        # Plot 1: Spatial spread map
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[0])
            im1 = axes[0].pcolormesh(lon_2d, lat_2d, metrics['spread'], 
                                   transform=ccrs.PlateCarree(), cmap='plasma')
        else:
            im1 = axes[0].imshow(metrics['spread'], cmap='plasma', origin='lower')
        
        axes[0].set_title(f'{var_name} - Ensemble Spread Map')
        plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Standard Deviation')
        
        # Add statistics text
        axes[0].text(0.02, 0.98, f'Mean Spread: {metrics["mean_spread"]:.4f}', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Spread distribution
        spread_flat = metrics['spread'].flatten()
        spread_valid = spread_flat[~np.isnan(spread_flat)]
        
        axes[1].hist(spread_valid, bins=30, alpha=0.7, edgecolor='black', color='orange')
        axes[1].axvline(metrics['mean_spread'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {metrics["mean_spread"]:.4f}')
        axes[1].axvline(np.median(spread_valid), color='blue', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(spread_valid):.4f}')
        axes[1].set_xlabel('Ensemble Spread (Std Dev)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{var_name} - Spread Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Multiple spread measures comparison
        if ground_truth is not None:
            # Spread-skill scatter plot
            error_flat = metrics['error'].flatten()
            error_valid = error_flat[~np.isnan(error_flat)]
            spread_for_scatter = spread_flat[~np.isnan(error_flat)]
            
            # Sample for cleaner visualization
            n_sample = min(1000, len(error_valid))
            if len(error_valid) > n_sample:
                indices = np.random.choice(len(error_valid), n_sample, replace=False)
                error_sample = error_valid[indices]
                spread_sample = spread_for_scatter[indices]
            else:
                error_sample = error_valid
                spread_sample = spread_for_scatter
            
            axes[2].scatter(spread_sample, error_sample, alpha=0.5, s=2)
            
            # Add diagonal line (perfect spread-skill relationship)
            min_val = min(np.min(spread_sample), np.min(error_sample))
            max_val = max(np.max(spread_sample), np.max(error_sample))
            axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', 
                        label='Perfect calibration')
            
            axes[2].set_xlabel('Ensemble Spread')
            axes[2].set_ylabel('Forecast Error')
            axes[2].set_title(f'{var_name} - Spread-Skill Relationship')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            # Add correlation text
            if not np.isnan(metrics['spread_skill_corr']):
                axes[2].text(0.05, 0.95, f'Correlation: {metrics["spread_skill_corr"]:.3f}', 
                            transform=axes[2].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Coefficient of variation map
            cv = metrics['coefficient_of_variation']
            cv_clipped = np.clip(cv, 0, np.percentile(cv[~np.isnan(cv)], 95))  # Clip outliers
            
            if self.config.USE_CARTOPY:
                self._setup_geographic_axes(axes[2])
                im3 = axes[2].pcolormesh(lon_2d, lat_2d, cv_clipped, 
                                       transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
            else:
                im3 = axes[2].imshow(cv_clipped, cmap='RdYlBu_r', origin='lower')
            
            axes[2].set_title(f'{var_name} - Coefficient of Variation')
            plt.colorbar(im3, ax=axes[2], shrink=0.8, label='CV (Spread/|Mean|)')
        
        # Plot 4: Summary statistics
        axes[3].axis('off')
        
        # Create summary statistics text
        if ground_truth is not None:
            stats_text = f"""Ensemble Spread Analysis

Spread Statistics:
  Mean Spread (σ): {metrics['mean_spread']:.4f}
  Mean IQR: {metrics['mean_iqr']:.4f}  
  Mean Range: {metrics['mean_range']:.4f}
  
Spread Percentiles:
  25th: {metrics['spread_percentiles'][0]:.4f}
  50th: {metrics['spread_percentiles'][1]:.4f}
  75th: {metrics['spread_percentiles'][2]:.4f}

Spread-Skill Relationship:
  Mean Error: {metrics['mean_error']:.4f}
  Spread/Skill Ratio: {metrics['spread_skill_ratio']:.3f}
  Correlation: {metrics['spread_skill_corr']:.3f}
  
Interpretation:
  Ratio ≈ 1: Well calibrated
  Ratio > 1: Overconfident  
  Ratio < 1: Underconfident"""
        else:
            stats_text = f"""Ensemble Spread Analysis

Spread Statistics:
  Mean Spread (σ): {metrics['mean_spread']:.4f}
  Mean Variance: {metrics['mean_variance']:.4f}
  Mean IQR: {metrics['mean_iqr']:.4f}  
  Mean Range: {metrics['mean_range']:.4f}
  
Spread Percentiles:
  25th: {metrics['spread_percentiles'][0]:.4f}
  50th: {metrics['spread_percentiles'][1]:.4f}
  75th: {metrics['spread_percentiles'][2]:.4f}

Note: Ground truth not available
for spread-skill analysis"""
        
        axes[3].text(0.1, 0.9, stats_text, transform=axes[3].transAxes, 
                    verticalalignment='top', fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Ensemble Spread Analysis (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"spread_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ Spread plot saved: {os.path.basename(output_path)}')
        
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