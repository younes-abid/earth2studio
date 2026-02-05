# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
PIT metric visualization - Probability Integral Transform analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from scipy import stats
from ..base_metric import BaseMetricPlot


class PITPlot(BaseMetricPlot):
    """
    Visualizes Probability Integral Transform (PIT) for ensemble forecasts.
    Assesses forecast calibration by examining the uniformity of PIT values.
    """
    
    def compute_metric(self, predictions, ground_truth, **kwargs):
        """
        Compute PIT values for ensemble predictions.
        
        Parameters
        ----------
        predictions : np.ndarray
            Ensemble predictions (ensemble_size, y, x)
        ground_truth : np.ndarray
            Ground truth values (y, x)
            
        Returns
        -------
        dict
            Dictionary containing PIT analysis results
        """
        bins = self.metric_params.get('bins', 10)
        
        # Flatten arrays for easier processing
        pred_flat = predictions.reshape(predictions.shape[0], -1)
        truth_flat = ground_truth.flatten()
        
        pit_values = []
        
        for i in range(pred_flat.shape[1]):
            # Get ensemble predictions for this point
            ensemble_point = pred_flat[:, i]
            truth_point = truth_flat[i]
            
            # Skip if truth is NaN
            if np.isnan(truth_point):
                continue
                
            # Sort ensemble members
            sorted_ensemble = np.sort(ensemble_point)
            
            # Find rank of truth in sorted ensemble
            # PIT = (rank + random uniform) / (ensemble_size + 1)
            rank = np.sum(sorted_ensemble <= truth_point)
            
            # Add random uniform for ties (standard PIT procedure)
            u = np.random.uniform(0, 1)
            pit = (rank + u) / (len(sorted_ensemble) + 1)
            pit_values.append(pit)
        
        pit_values = np.array(pit_values)
        
        # Create histogram of PIT values
        hist, bin_edges = np.histogram(pit_values, bins=bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Expected frequency for uniform distribution
        expected_freq = len(pit_values) / bins
        
        # Chi-square test for uniformity
        chi_square = np.sum((hist - expected_freq) ** 2 / expected_freq) if expected_freq > 0 else np.nan
        chi_square_p_value = 1 - stats.chi2.cdf(chi_square, bins - 1) if not np.isnan(chi_square) else np.nan
        
        # Kolmogorov-Smirnov test statistic
        # Compare empirical CDF to uniform CDF
        sorted_pit = np.sort(pit_values)
        uniform_cdf = np.arange(1, len(sorted_pit) + 1) / len(sorted_pit)
        ks_statistic = np.max(np.abs(sorted_pit - uniform_cdf))
        
        # Anderson-Darling test for uniformity
        n = len(pit_values)
        if n > 0:
            ad_statistic = -n - np.sum((2 * np.arange(1, n + 1) - 1) * 
                                     (np.log(sorted_pit) + np.log(1 - sorted_pit[::-1]))) / n
        else:
            ad_statistic = np.nan
        
        # Compute bins for reliability analysis
        alpha_levels = [0.1, 0.05]  # 90% and 95% confidence levels
        reliability_results = {}
        
        for alpha in alpha_levels:
            # Expected number of PIT values in each tail
            expected_in_tail = alpha * len(pit_values) / 2
            actual_left_tail = np.sum(pit_values < alpha/2)
            actual_right_tail = np.sum(pit_values > 1 - alpha/2)
            
            reliability_results[f'alpha_{alpha}'] = {
                'expected_per_tail': expected_in_tail,
                'actual_left_tail': actual_left_tail,
                'actual_right_tail': actual_right_tail,
                'total_expected': alpha * len(pit_values),
                'total_actual': actual_left_tail + actual_right_tail
            }
        
        return {
            'pit_values': pit_values,
            'histogram': hist,
            'bin_centers': bin_centers,
            'bin_edges': bin_edges,
            'expected_frequency': expected_freq,
            'chi_square': chi_square,
            'chi_square_p_value': chi_square_p_value,
            'ks_statistic': ks_statistic,
            'ad_statistic': ad_statistic,
            'uniformity_score': 1 / (1 + ks_statistic),  # Simple uniformity measure
            'sorted_pit': sorted_pit,
            'reliability_results': reliability_results,
            'n_samples': len(pit_values)
        }
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate PIT histogram and uniformity analysis plots.
        
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
        print(f'🎨 Creating PIT analysis plots for time index {time_idx}...')
        
        # Try to load ground truth data
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                var_name = self.config.OUTPUT_VARIABLES[variable_idx]
                if var_name not in truth_ds.data_vars:
                    print(f'   ⚠️  Ground truth for {var_name} not available, skipping PIT analysis')
                    return None
                ground_truth = truth_ds[var_name].isel(time=time_idx).values
        except (OSError, KeyError):
            print(f'   ⚠️  Ground truth data not available, skipping PIT analysis')
            return None
        
        # Load prediction data
        with xr.open_dataset(netcdf_path, group='prediction') as ds:
            var_name = self.config.OUTPUT_VARIABLES[variable_idx]
            predictions = ds[var_name].isel(time=time_idx).values  # Shape: (ensemble, y, x)
        
        # Compute PIT metrics
        metrics = self.compute_metric(predictions, ground_truth)
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: PIT histogram with uniform reference
        axes[0, 0].bar(metrics['bin_centers'], metrics['histogram'], 
                      width=1/len(metrics['bin_centers']), alpha=0.7, 
                      color='skyblue', edgecolor='black', label='Observed')
        
        # Add expected uniform frequency line
        axes[0, 0].axhline(y=metrics['expected_frequency'], color='red', 
                          linestyle='--', linewidth=2, label='Expected (Uniform)')
        
        # Add confidence bands for uniform distribution (95%)
        n = metrics['n_samples']
        std_uniform = np.sqrt(n / len(metrics['bin_centers'])) / 2  # Standard error
        conf_band = 1.96 * std_uniform  # 95% confidence interval
        axes[0, 0].fill_between([0, 1], 
                               metrics['expected_frequency'] - conf_band,
                               metrics['expected_frequency'] + conf_band,
                               alpha=0.3, color='red', label='95% Conf. Band')
        
        axes[0, 0].set_xlabel('PIT Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'{var_name} - PIT Histogram')
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add test statistics
        axes[0, 0].text(0.02, 0.98, f'χ² = {metrics["chi_square"]:.2f}\np = {metrics["chi_square_p_value"]:.3f}', 
                       transform=axes[0, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Q-Q plot against uniform distribution
        uniform_quantiles = np.linspace(0, 1, len(metrics['sorted_pit']))
        
        axes[0, 1].scatter(uniform_quantiles, metrics['sorted_pit'], alpha=0.6, s=2)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Uniform')
        
        # Add confidence bands
        n = len(metrics['sorted_pit'])
        se = 1.36 / np.sqrt(n)  # Standard error for uniform Q-Q plot
        axes[0, 1].fill_between([0, 1], [0 - se, 1 - se], [0 + se, 1 + se], 
                               alpha=0.3, color='gray', label='95% Conf. Band')
        
        axes[0, 1].set_xlabel('Theoretical Uniform Quantiles')
        axes[0, 1].set_ylabel('Observed PIT Quantiles')
        axes[0, 1].set_title(f'{var_name} - Q-Q Plot vs Uniform')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add KS statistic
        axes[0, 1].text(0.02, 0.98, f'KS = {metrics["ks_statistic"]:.3f}', 
                       transform=axes[0, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Empirical CDF vs Uniform CDF
        x_vals = np.linspace(0, 1, 100)
        empirical_cdf = np.searchsorted(metrics['sorted_pit'], x_vals, side='right') / len(metrics['sorted_pit'])
        
        axes[1, 0].plot(x_vals, empirical_cdf, label='Empirical CDF', linewidth=2)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Uniform CDF', linewidth=2)
        axes[1, 0].fill_between(x_vals, empirical_cdf, x_vals, alpha=0.3, 
                               where=(empirical_cdf > x_vals), color='red', label='Above Uniform')
        axes[1, 0].fill_between(x_vals, empirical_cdf, x_vals, alpha=0.3, 
                               where=(empirical_cdf < x_vals), color='blue', label='Below Uniform')
        
        axes[1, 0].set_xlabel('PIT Value')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title(f'{var_name} - Empirical vs Uniform CDF')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics and interpretation
        axes[1, 1].axis('off')
        
        # Create summary statistics text
        stats_text = f"""PIT Analysis Summary

Sample Size: {metrics['n_samples']:,} points

Uniformity Tests:
  Chi-Square: {metrics['chi_square']:.3f}
  p-value: {metrics['chi_square_p_value']:.3f}
  KS Statistic: {metrics['ks_statistic']:.3f}
  AD Statistic: {metrics['ad_statistic']:.3f}
  Uniformity Score: {metrics['uniformity_score']:.3f}

Reliability Analysis:"""
        
        for alpha_key, rel_result in metrics['reliability_results'].items():
            alpha = float(alpha_key.split('_')[1])
            stats_text += f"""
  {(1-alpha)*100:.0f}% Level:
    Expected in tails: {rel_result['total_expected']:.1f}
    Actual in tails: {rel_result['total_actual']}
    Left tail: {rel_result['actual_left_tail']} (exp: {rel_result['expected_per_tail']:.1f})
    Right tail: {rel_result['actual_right_tail']} (exp: {rel_result['expected_per_tail']:.1f})"""
        
        stats_text += f"""

Interpretation:
  p > 0.05: Consistent with uniform
  p ≤ 0.05: Significantly non-uniform
  KS < 0.1: Good calibration
  Flat histogram: Well-calibrated
  U-shape: Overconfident
  Inverse U: Underconfident"""
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontsize=9, fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Probability Integral Transform (PIT) Analysis (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"pit_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ PIT plot saved: {os.path.basename(output_path)}')
        
        return output_path