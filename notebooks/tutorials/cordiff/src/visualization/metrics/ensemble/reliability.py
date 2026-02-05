# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Reliability metric visualization - Ensemble forecast calibration analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from scipy import stats
from ..base_metric import BaseMetricPlot


class ReliabilityPlot(BaseMetricPlot):
    """
    Visualizes reliability (calibration) for ensemble forecasts.
    Compares predicted probabilities with observed frequencies.
    """
    
    def compute_metric(self, predictions, ground_truth, **kwargs):
        """
        Compute reliability diagram for ensemble predictions.
        
        Parameters
        ----------
        predictions : np.ndarray
            Ensemble predictions (ensemble_size, y, x)
        ground_truth : np.ndarray
            Ground truth values (y, x)
            
        Returns
        -------
        dict
            Dictionary containing reliability analysis results
        """
        bins = self.metric_params.get('bins', 10)
        
        # Flatten arrays for easier processing
        pred_flat = predictions.reshape(predictions.shape[0], -1)
        truth_flat = ground_truth.flatten()
        
        # Define multiple thresholds for analysis
        percentiles = self.metric_params.get('percentiles', [10, 25, 50, 75, 90])
        valid_truth = truth_flat[~np.isnan(truth_flat)]
        thresholds = [np.percentile(valid_truth, p) for p in percentiles]
        
        reliability_results = {}
        
        for i, (threshold, percentile) in enumerate(zip(thresholds, percentiles)):
            # Convert to binary event: exceeding threshold
            binary_truth = (truth_flat > threshold).astype(float)
            
            # Compute ensemble probability of exceeding threshold
            prob_exceed = np.mean(pred_flat > threshold, axis=0)
            
            # Remove NaN values
            valid_mask = ~(np.isnan(binary_truth) | np.isnan(prob_exceed))
            binary_truth_valid = binary_truth[valid_mask]
            prob_exceed_valid = prob_exceed[valid_mask]
            
            if len(binary_truth_valid) == 0:
                continue
            
            # Create probability bins
            bin_edges = np.linspace(0, 1, bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Assign each forecast to a probability bin
            bin_indices = np.digitize(prob_exceed_valid, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, bins - 1)
            
            # Compute reliability for each bin
            observed_freq = np.full(bins, np.nan)
            forecast_prob = np.full(bins, np.nan)
            bin_counts = np.zeros(bins)
            
            for b in range(bins):
                mask = (bin_indices == b)
                if np.sum(mask) > 0:
                    observed_freq[b] = np.mean(binary_truth_valid[mask])
                    forecast_prob[b] = np.mean(prob_exceed_valid[mask])
                    bin_counts[b] = np.sum(mask)
                else:
                    forecast_prob[b] = bin_centers[b]
            
            # Compute reliability score (closer to 0 is better)
            valid_bins = ~np.isnan(observed_freq)
            if np.sum(valid_bins) > 0:
                reliability_score = np.sqrt(np.sum(
                    bin_counts[valid_bins] * (observed_freq[valid_bins] - forecast_prob[valid_bins]) ** 2
                ) / np.sum(bin_counts[valid_bins]))
            else:
                reliability_score = np.nan
            
            # Compute resolution (ability to distinguish events)
            base_rate = np.mean(binary_truth_valid)
            if np.sum(valid_bins) > 0:
                resolution = np.sum(bin_counts[valid_bins] * (observed_freq[valid_bins] - base_rate) ** 2) / len(binary_truth_valid)
            else:
                resolution = np.nan
            
            # Compute uncertainty (inherent variability)
            uncertainty = base_rate * (1 - base_rate)
            
            # Brier score and its decomposition
            brier_score = np.mean((prob_exceed_valid - binary_truth_valid) ** 2)
            
            # Brier skill score
            brier_skill_score = 1 - brier_score / uncertainty if uncertainty > 0 else np.nan
            
            # Slope and intercept of reliability line (linear fit)
            valid_for_fit = valid_bins & (bin_counts > 0)
            if np.sum(valid_for_fit) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    forecast_prob[valid_for_fit], observed_freq[valid_for_fit])
                reliability_slope = slope
                reliability_intercept = intercept
                reliability_r2 = r_value ** 2
            else:
                reliability_slope = np.nan
                reliability_intercept = np.nan
                reliability_r2 = np.nan
            
            reliability_results[f'threshold_{percentile}p'] = {
                'threshold': threshold,
                'threshold_percentile': percentile,
                'bin_centers': bin_centers,
                'bin_edges': bin_edges,
                'observed_frequency': observed_freq,
                'forecast_probability': forecast_prob,
                'bin_counts': bin_counts,
                'reliability_score': reliability_score,
                'resolution': resolution,
                'uncertainty': uncertainty,
                'brier_score': brier_score,
                'base_rate': base_rate,
                'brier_skill_score': brier_skill_score,
                'reliability_slope': reliability_slope,
                'reliability_intercept': reliability_intercept,
                'reliability_r2': reliability_r2,
                'binary_truth': binary_truth_valid,
                'prob_exceed': prob_exceed_valid
            }
        
        return reliability_results
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate reliability diagram plots.
        
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
        print(f'🎨 Creating reliability diagram plots for time index {time_idx}...')
        
        # Try to load ground truth data
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                var_name = self.config.OUTPUT_VARIABLES[variable_idx]
                if var_name not in truth_ds.data_vars:
                    print(f'   ⚠️  Ground truth for {var_name} not available, skipping reliability analysis')
                    return None
                ground_truth = truth_ds[var_name].isel(time=time_idx).values
        except (OSError, KeyError):
            print(f'   ⚠️  Ground truth data not available, skipping reliability analysis')
            return None
        
        # Load prediction data
        with xr.open_dataset(netcdf_path, group='prediction') as ds:
            var_name = self.config.OUTPUT_VARIABLES[variable_idx]
            predictions = ds[var_name].isel(time=time_idx).values  # Shape: (ensemble, y, x)
        
        # Compute reliability metrics
        metrics = self.compute_metric(predictions, ground_truth)
        
        if not metrics:
            print(f'   ⚠️  Could not compute reliability metrics')
            return None
        
        # Create subplot layout - show 2x2 grid with selected thresholds
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Select up to 4 thresholds for detailed display
        threshold_keys = list(metrics.keys())[:4]
        colors = ['blue', 'green', 'orange', 'red']
        
        # Plot 1: Reliability diagrams for multiple thresholds
        for i, (key, color) in enumerate(zip(threshold_keys, colors)):
            metric = metrics[key]
            valid_bins = ~np.isnan(metric['observed_frequency'])
            
            if np.sum(valid_bins) > 0:
                axes[0, 0].plot(metric['forecast_probability'][valid_bins], 
                               metric['observed_frequency'][valid_bins], 
                               'o-', color=color, label=f"{metric['threshold_percentile']}p", 
                               markersize=6, linewidth=2)
        
        # Perfect reliability line
        axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Reliability')
        
        # No skill line (base rate)
        if threshold_keys:
            base_rates = [metrics[key]['base_rate'] for key in threshold_keys]
            avg_base_rate = np.mean([br for br in base_rates if not np.isnan(br)])
            if not np.isnan(avg_base_rate):
                axes[0, 0].axhline(y=avg_base_rate, color='gray', linestyle=':', 
                                  label=f'Base Rate ({avg_base_rate:.2f})')
        
        axes[0, 0].set_xlabel('Forecast Probability')
        axes[0, 0].set_ylabel('Observed Frequency')
        axes[0, 0].set_title(f'{var_name} - Reliability Diagrams')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Forecast probability histograms
        for key, color in zip(threshold_keys, colors):
            metric = metrics[key]
            axes[0, 1].hist(metric['prob_exceed'], bins=20, alpha=0.5, 
                           color=color, label=f"{metric['threshold_percentile']}p", 
                           density=True)
        
        axes[0, 1].set_xlabel('Forecast Probability')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title(f'{var_name} - Probability Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Brier score decomposition
        if len(threshold_keys) > 0:
            threshold_labels = [f"{metrics[key]['threshold_percentile']}p" for key in threshold_keys]
            reliability_scores = [metrics[key]['reliability_score'] for key in threshold_keys]
            resolution_scores = [metrics[key]['resolution'] for key in threshold_keys]
            brier_scores = [metrics[key]['brier_score'] for key in threshold_keys]
            
            x = np.arange(len(threshold_labels))
            width = 0.25
            
            axes[1, 0].bar(x - width, reliability_scores, width, label='Reliability', 
                          color='red', alpha=0.7)
            axes[1, 0].bar(x, resolution_scores, width, label='Resolution', 
                          color='blue', alpha=0.7)
            axes[1, 0].bar(x + width, brier_scores, width, label='Brier Score', 
                          color='green', alpha=0.7)
            
            axes[1, 0].set_xlabel('Threshold')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title(f'{var_name} - Brier Score Decomposition')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(threshold_labels)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        axes[1, 1].axis('off')
        
        # Create summary statistics text
        stats_text = f"""Reliability Analysis Summary

Thresholds Analyzed:"""
        
        for key in threshold_keys:
            metric = metrics[key]
            stats_text += f"""
  {metric['threshold_percentile']}th percentile:
    Value: {metric['threshold']:.3f}
    Base Rate: {metric['base_rate']:.3f}
    Reliability: {metric['reliability_score']:.3f}
    Resolution: {metric['resolution']:.3f}
    Brier Score: {metric['brier_score']:.3f}
    Brier Skill: {metric['brier_skill_score']:.3f}
    R²: {metric['reliability_r2']:.3f}"""
        
        stats_text += f"""

Interpretation:
  Reliability: Lower is better (0 = perfect)
  Resolution: Higher is better
  Brier Score: Lower is better (0 = perfect)
  Brier Skill: Higher is better (1 = perfect)
  
Quality Guidelines:
  Points on diagonal: Well calibrated
  Above diagonal: Underconfident
  Below diagonal: Overconfident"""
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontsize=9, fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Ensemble Forecast Reliability Analysis (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"reliability_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ Reliability plot saved: {os.path.basename(output_path)}')
        
        return output_path