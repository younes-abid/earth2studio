# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
R2 metric visualization - Coefficient of Determination.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base_metric import BaseMetricPlot


class R2Plot(BaseMetricPlot):
    """
    Visualizes R-squared (Coefficient of Determination) metric.
    """
    
    def compute_metric(self, predictions, ground_truth, **kwargs):
        """
        Compute R² between predictions and ground truth.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions (ensemble, y, x)
        ground_truth : np.ndarray
            Ground truth values (y, x)
            
        Returns
        -------
        dict
            Dictionary with R² values
        """
        # Calculate ensemble mean
        ensemble_mean = np.mean(predictions, axis=0)
        
        # Flatten arrays for correlation calculation
        pred_flat = ensemble_mean.flatten()
        truth_flat = ground_truth.flatten()
        
        # Remove NaN values and infinite values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(truth_flat) | 
                      np.isinf(pred_flat) | np.isinf(truth_flat))
        pred_valid = pred_flat[valid_mask]
        truth_valid = truth_flat[valid_mask]
        
        if len(pred_valid) == 0:
            print("   ⚠️  Warning: No valid data points for R² calculation")
            return self._return_nan_metrics()
        
        # Check for constant ground truth (would make R² undefined)
        if np.var(truth_valid) < 1e-10:
            print("   ⚠️  Warning: Ground truth is nearly constant, R² undefined")
            return self._return_nan_metrics()
        
        # Compute global R² with improved numerical stability
        truth_mean = np.mean(truth_valid)
        ss_tot = np.sum((truth_valid - truth_mean) ** 2)
        ss_res = np.sum((truth_valid - pred_valid) ** 2)
        
        # Add small epsilon to prevent division issues
        eps = 1e-10
        r2_global = 1 - (ss_res / (ss_tot + eps))
        
        # Clip extreme R² values for better visualization
        r2_global_clipped = max(r2_global, -100.0)  # Don't allow R² below -100
        
        # Compute correlation coefficient
        correlation = np.corrcoef(pred_valid, truth_valid)[0, 1] if len(pred_valid) > 1 else 0
        
        # Compute R² for each ensemble member
        r2_members = []
        for i in range(predictions.shape[0]):
            pred_member = predictions[i].flatten()
            pred_member_valid = pred_member[valid_mask]
            ss_res_member = np.sum((truth_valid - pred_member_valid) ** 2)
            r2_member = 1 - (ss_res_member / (ss_tot + eps))
            r2_member_clipped = max(r2_member, -100.0)  # Clip extreme values
            r2_members.append(r2_member_clipped)
        
        r2_members = np.array(r2_members)
        
        # Calculate additional diagnostic metrics
        rmse_global = np.sqrt(np.mean((truth_valid - pred_valid) ** 2))
        mae_global = np.mean(np.abs(truth_valid - pred_valid))
        bias_global = np.mean(pred_valid - truth_valid)
        
        # Calculate baseline metrics (using mean as predictor)
        baseline_pred = np.full_like(truth_valid, truth_mean)
        rmse_baseline = np.sqrt(np.mean((truth_valid - baseline_pred) ** 2))
        
        return {
            'r2_global': r2_global,
            'r2_global_clipped': r2_global_clipped,
            'r2_members': r2_members,
            'r2_std': np.std(r2_members),
            'correlation': correlation,
            'rmse_global': rmse_global,
            'rmse_baseline': rmse_baseline,
            'mae_global': mae_global,
            'bias_global': bias_global,
            'ensemble_mean': ensemble_mean,
            'ground_truth': ground_truth,
            'pred_valid': pred_valid,
            'truth_valid': truth_valid,
            'num_valid_points': len(pred_valid),
            'data_range': {
                'truth_min': np.min(truth_valid),
                'truth_max': np.max(truth_valid),
                'pred_min': np.min(pred_valid),
                'pred_max': np.max(pred_valid)
            }
        }
    
    def _return_nan_metrics(self):
        """Return dictionary with NaN values when computation fails."""
        return {
            'r2_global': np.nan,
            'r2_global_clipped': np.nan,
            'r2_members': np.array([np.nan]),
            'r2_std': np.nan,
            'correlation': np.nan,
            'rmse_global': np.nan,
            'rmse_baseline': np.nan,
            'mae_global': np.nan,
            'bias_global': np.nan,
            'ensemble_mean': None,
            'ground_truth': None,
            'pred_valid': np.array([]),
            'truth_valid': np.array([]),
            'num_valid_points': 0,
            'data_range': {}
        }
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate R² visualization plots.
        
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
        print(f'🎨 Creating R² metric plots for time index {time_idx}...')
        
        # Try to load ground truth data
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                var_name = self.config.OUTPUT_VARIABLES[variable_idx]
                if var_name not in truth_ds.data_vars:
                    print(f'   ⚠️  Ground truth for {var_name} not available, skipping R²')
                    return None
                ground_truth = truth_ds[var_name].isel(time=time_idx).values
        except (OSError, KeyError):
            print(f'   ⚠️  Ground truth data not available, skipping R²')
            return None
        
        # Load prediction data
        with xr.open_dataset(netcdf_path, group='prediction') as ds:
            var_name = self.config.OUTPUT_VARIABLES[variable_idx]
            predictions = ds[var_name].isel(time=time_idx).values  # Shape: (ensemble, y, x)
        
        # Compute R² metrics
        metrics = self.compute_metric(predictions, ground_truth)
        
        # Create subplot layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Scatter plot of predictions vs ground truth
        axes[0].scatter(metrics['truth_valid'], metrics['pred_valid'], 
                       alpha=0.6, s=1, c='blue')
        
        # Add perfect correlation line
        min_val = min(metrics['truth_valid'].min(), metrics['pred_valid'].min())
        max_val = max(metrics['truth_valid'].max(), metrics['pred_valid'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', 
                    label='Perfect correlation')
        
        # Add regression line
        z = np.polyfit(metrics['truth_valid'], metrics['pred_valid'], 1)
        p = np.poly1d(z)
        axes[0].plot(metrics['truth_valid'], p(metrics['truth_valid']), 
                    'g-', linewidth=2, label=f'Regression line')
        
        axes[0].set_xlabel('Ground Truth')
        axes[0].set_ylabel('Predictions (Ensemble Mean)')
        axes[0].set_title(f'{var_name} - Predictions vs Ground Truth')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add R² annotation with more diagnostic info
        r2_display = metrics["r2_global_clipped"] if abs(metrics["r2_global"]) > 100 else metrics["r2_global"]
        annotation_text = (f'R² = {r2_display:.4f}\n'
                          f'r = {metrics["correlation"]:.4f}\n'
                          f'RMSE = {metrics["rmse_global"]:.4f}\n'
                          f'Bias = {metrics["bias_global"]:.4f}')
        
        if abs(metrics["r2_global"]) > 100:
            annotation_text += f'\n(R² clipped from {metrics["r2_global"]:.1e})'
            
        axes[0].text(0.05, 0.95, annotation_text, 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: R² distribution across ensemble members
        axes[1].hist(metrics['r2_members'], bins=min(10, len(metrics['r2_members'])), 
                    alpha=0.7, edgecolor='black', color='lightgreen')
        axes[1].axvline(metrics['r2_global_clipped'], color='red', linestyle='--', linewidth=2, 
                       label=f'Ensemble Mean R²: {metrics["r2_global_clipped"]:.4f}')
        axes[1].set_xlabel('R² Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{var_name} - R² Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add summary statistics with improved formatting
        r2_members_mean = np.mean(metrics['r2_members'])
        stats_text = (f'Members Mean: {r2_members_mean:.4f}\n'
                     f'Ensemble R²: {metrics["r2_global_clipped"]:.4f}\n'
                     f'Std: {metrics["r2_std"]:.4f}\n'
                     f'Min: {np.min(metrics["r2_members"]):.4f}\n'
                     f'Max: {np.max(metrics["r2_members"]):.4f}\n'
                     f'Valid points: {metrics["num_valid_points"]}')
        axes[1].text(0.98, 0.98, stats_text, transform=axes[1].transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: R² values for each ensemble member
        member_indices = np.arange(len(metrics['r2_members']))
        axes[2].bar(member_indices, metrics['r2_members'], alpha=0.7, color='lightblue')
        axes[2].axhline(metrics['r2_global_clipped'], color='red', linestyle='--', linewidth=2,
                       label=f'Ensemble Mean R²: {metrics["r2_global_clipped"]:.4f}')
        axes[2].set_xlabel('Ensemble Member')
        axes[2].set_ylabel('R² Value')
        axes[2].set_title(f'{var_name} - R² by Ensemble Member')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'{var_name} - Coefficient of Determination Analysis (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"r2_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ R² plot saved: {os.path.basename(output_path)}')
        
        # Print diagnostic information to console
        print(f'   📊 R² Analysis Summary:')
        print(f'      Raw R²: {metrics["r2_global"]:.6f}')
        print(f'      Clipped R²: {metrics["r2_global_clipped"]:.6f}')
        print(f'      Correlation: {metrics["correlation"]:.6f}')
        print(f'      RMSE: {metrics["rmse_global"]:.6f}')
        print(f'      Baseline RMSE: {metrics["rmse_baseline"]:.6f}')
        print(f'      MAE: {metrics["mae_global"]:.6f}')
        print(f'      Bias: {metrics["bias_global"]:.6f}')
        print(f'      Data range - Truth: [{metrics["data_range"]["truth_min"]:.3f}, {metrics["data_range"]["truth_max"]:.3f}]')
        print(f'      Data range - Pred: [{metrics["data_range"]["pred_min"]:.3f}, {metrics["data_range"]["pred_max"]:.3f}]')
        
        if metrics["r2_global"] < -10:
            print(f'   ⚠️  Very low R² indicates poor model performance - predictions much worse than using mean')
        elif metrics["r2_global"] < 0:
            print(f'   ⚠️  Negative R² indicates predictions worse than using mean as baseline')
        
        return output_path