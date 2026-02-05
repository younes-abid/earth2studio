# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Coverage metric visualization - Prediction interval coverage analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base_metric import BaseMetricPlot


class CoveragePlot(BaseMetricPlot):
    """
    Visualizes prediction interval coverage for ensemble forecasts.
    Assesses whether confidence intervals contain the true values at expected rates.
    """
    
    def compute_metric(self, predictions, ground_truth, **kwargs):
        """
        Compute prediction interval coverage for ensemble predictions.
        
        Parameters
        ----------
        predictions : np.ndarray
            Ensemble predictions (ensemble_size, y, x)
        ground_truth : np.ndarray
            Ground truth values (y, x)
            
        Returns
        -------
        dict
            Dictionary containing coverage metrics for each confidence level
        """
        confidence_levels = self.metric_params.get('confidence_levels', [0.5, 0.68, 0.9, 0.95])
        
        coverage_results = {}
        spatial_coverage = {}
        
        for conf_level in confidence_levels:
            # Calculate percentiles for prediction intervals
            lower_percentile = (1 - conf_level) / 2 * 100
            upper_percentile = (1 + conf_level) / 2 * 100
            
            # Compute prediction intervals
            lower_bound = np.percentile(predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(predictions, upper_percentile, axis=0)
            
            # Check if ground truth falls within intervals
            within_interval = (ground_truth >= lower_bound) & (ground_truth <= upper_bound)
            
            # Calculate coverage percentage (global)
            coverage = np.mean(within_interval) * 100
            
            # Calculate interval width (sharpness)
            interval_width = upper_bound - lower_bound
            mean_interval_width = np.mean(interval_width)
            
            # Normalized interval width (by ensemble standard deviation)
            ensemble_std = np.std(predictions, axis=0)
            normalized_width = interval_width / (ensemble_std + 1e-8)
            
            # Coverage by spatial location (for mapping)
            spatial_coverage[f'{conf_level:.0%}'] = within_interval.astype(float)
            
            coverage_results[f'{conf_level:.0%}'] = {
                'confidence_level': conf_level,
                'expected_coverage': conf_level * 100,
                'actual_coverage': coverage,
                'coverage_difference': coverage - (conf_level * 100),
                'interval_width': interval_width,
                'mean_interval_width': mean_interval_width,
                'normalized_width': normalized_width,
                'mean_normalized_width': np.mean(normalized_width),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'within_interval': within_interval,
                'width_std': np.std(interval_width)
            }
        
        # Overall coverage statistics
        overall_stats = {
            'spatial_coverage': spatial_coverage,
            'ensemble_mean': np.mean(predictions, axis=0),
            'ground_truth': ground_truth
        }
        
        coverage_results['overall'] = overall_stats
        
        return coverage_results
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate prediction interval coverage visualization plots.
        
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
        print(f'🎨 Creating coverage analysis plots for time index {time_idx}...')
        
        # Try to load ground truth data
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                var_name = self.config.OUTPUT_VARIABLES[variable_idx]
                if var_name not in truth_ds.data_vars:
                    print(f'   ⚠️  Ground truth for {var_name} not available, skipping coverage analysis')
                    return None
                ground_truth = truth_ds[var_name].isel(time=time_idx).values
        except (OSError, KeyError):
            print(f'   ⚠️  Ground truth data not available, skipping coverage analysis')
            return None
        
        # Load prediction data and coordinates
        with xr.open_dataset(netcdf_path, group='prediction') as ds:
            var_name = self.config.OUTPUT_VARIABLES[variable_idx]
            predictions = ds[var_name].isel(time=time_idx).values  # Shape: (ensemble, y, x)
            
            # Get coordinates
            lat_2d, lon_2d = self._get_coordinates(ds)
        
        # Compute coverage metrics
        metrics = self.compute_metric(predictions, ground_truth)
        
        # Create subplot layout
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Coverage vs Expected Coverage
        confidence_levels = [key for key in metrics.keys() if key != 'overall']
        expected_cov = [metrics[key]['expected_coverage'] for key in confidence_levels]
        actual_cov = [metrics[key]['actual_coverage'] for key in confidence_levels]
        coverage_diff = [metrics[key]['coverage_difference'] for key in confidence_levels]
        
        axes[0, 0].plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect calibration')
        bars = axes[0, 0].bar(range(len(confidence_levels)), actual_cov, 
                             alpha=0.7, color=['skyblue', 'lightgreen', 'orange', 'coral'])
        
        # Add expected coverage as horizontal lines
        for i, (exp, act) in enumerate(zip(expected_cov, actual_cov)):
            axes[0, 0].axhline(y=exp, xmin=i/len(confidence_levels) + 0.1/len(confidence_levels), 
                              xmax=(i+1)/len(confidence_levels) - 0.1/len(confidence_levels), 
                              color='red', linestyle='--', linewidth=2)
        
        axes[0, 0].set_xlabel('Confidence Level')
        axes[0, 0].set_ylabel('Coverage (%)')
        axes[0, 0].set_title(f'{var_name} - Coverage vs Expected')
        axes[0, 0].set_xticks(range(len(confidence_levels)))
        axes[0, 0].set_xticklabels(confidence_levels)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add coverage difference as text
        for i, (bar, diff) in enumerate(zip(bars, coverage_diff)):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{diff:+.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Spatial coverage map for 90% confidence level
        coverage_90 = metrics.get('90%', metrics.get('95%', None))
        if coverage_90 is not None:
            if self.config.USE_CARTOPY:
                ax_map = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
                self._setup_geographic_axes(ax_map)
                im2 = ax_map.pcolormesh(lon_2d, lat_2d, coverage_90['within_interval'].astype(float), 
                                       transform=ccrs.PlateCarree(), cmap='RdYlGn', vmin=0, vmax=1)
            else:
                im2 = axes[0, 1].imshow(coverage_90['within_interval'].astype(float), 
                                       cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
                ax_map = axes[0, 1]
            
            ax_map.set_title(f'{var_name} - Spatial Coverage (90%)')
            plt.colorbar(im2, ax=ax_map, shrink=0.8, label='Within Interval (1=Yes, 0=No)')
        
        # Plot 3: Interval width vs coverage trade-off
        mean_widths = [metrics[key]['mean_interval_width'] for key in confidence_levels]
        
        axes[1, 0].scatter(mean_widths, actual_cov, s=100, alpha=0.7, c=expected_cov, cmap='viridis')
        
        # Add labels for each point
        for i, (width, cov, level) in enumerate(zip(mean_widths, actual_cov, confidence_levels)):
            axes[1, 0].annotate(level, (width, cov), xytext=(5, 5), 
                               textcoords='offset points', fontsize=9)
        
        axes[1, 0].set_xlabel('Mean Interval Width')
        axes[1, 0].set_ylabel('Coverage (%)')
        axes[1, 0].set_title(f'{var_name} - Width vs Coverage Trade-off')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics and interval example
        axes[1, 1].axis('off')
        
        # Create summary statistics text
        stats_text = f"""Coverage Analysis Summary

Confidence Levels & Performance:"""
        
        for level in confidence_levels:
            metric = metrics[level]
            stats_text += f"""
  {level}: {metric['actual_coverage']:.1f}% (exp: {metric['expected_coverage']:.0f}%)
    Difference: {metric['coverage_difference']:+.1f}%
    Mean Width: {metric['mean_interval_width']:.3f}"""
        
        stats_text += f"""

Quality Assessment:
  Well-calibrated: Actual ≈ Expected
  Over-confident: Actual < Expected
  Under-confident: Actual > Expected
  
Sharpness (smaller is better):
  Normalized width shows interval
  efficiency relative to spread"""
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontsize=9, fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Prediction Interval Coverage Analysis (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"coverage_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ Coverage plot saved: {os.path.basename(output_path)}')
        
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