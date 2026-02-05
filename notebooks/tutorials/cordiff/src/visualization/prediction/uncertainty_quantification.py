# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Uncertainty quantification visualization - Prediction intervals and reliability analysis.
Corresponds to config: PLOT_UNCERTAINTY_QUANTIFICATION = True
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base import BasePlot


class UncertaintyQuantificationPlot(BasePlot):
    """
    Creates uncertainty quantification plots with prediction intervals and reliability.
    """
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, confidence_levels=None, **kwargs):
        """
        Generate uncertainty quantification plots.
        
        Parameters
        ----------
        netcdf_path : str
            Path to NetCDF ensemble file
        time_idx : int
            Time index to plot
        variable_idx : int
            Variable index to plot (if multiple variables)
        confidence_levels : list, optional
            Confidence levels for prediction intervals (default: [0.5, 0.68, 0.9, 0.95])
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        str
            Path to saved plot file
        """
        if confidence_levels is None:
            confidence_levels = [0.5, 0.68, 0.9, 0.95]
            
        print(f'🎨 Creating uncertainty quantification plots for time index {time_idx}...')
        
        # Load prediction data
        with xr.open_dataset(netcdf_path, group='prediction') as ds:
            var_name = self.config.OUTPUT_VARIABLES[variable_idx]
            predictions = ds[var_name].isel(time=time_idx)  # Shape: (ensemble, y, x)
            
            # Get coordinates - check both coords and data_vars
            lat_2d = None
            lon_2d = None
            
            # Try to get coordinates from proper coords first
            if 'lat' in ds.coords and 'lon' in ds.coords:
                if ds.coords['lat'].ndim == 2:  # 2D coordinates
                    lat_2d = ds.coords['lat'].values
                    lon_2d = ds.coords['lon'].values
                else:  # 1D coordinates
                    lat_1d = ds.coords['lat'].values
                    lon_1d = ds.coords['lon'].values
                    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
            
            # If not found in coords, try data_vars (our current case)
            elif 'lat' in ds.data_vars and 'lon' in ds.data_vars:
                lat_data = ds['lat'].values
                lon_data = ds['lon'].values
                
                if lat_data.ndim == 2:  # 2D coordinates
                    lat_2d = lat_data
                    lon_2d = lon_data
                else:  # 1D coordinates
                    lon_2d, lat_2d = np.meshgrid(lon_data, lat_data)
            
            # Final fallback to config grid
            else:
                lat_1d = np.linspace(*self.config.OUTPUT_GRID['lat'])
                lon_1d = np.linspace(*self.config.OUTPUT_GRID['lon'])
                lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        
        # Calculate ensemble statistics
        ensemble_mean = predictions.mean(dim='ensemble')
        ensemble_std = predictions.std(dim='ensemble')
        
        # Calculate prediction intervals (quantiles)
        prediction_intervals = {}
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2
            
            lower = predictions.quantile(lower_quantile, dim='ensemble')
            upper = predictions.quantile(upper_quantile, dim='ensemble')
            interval_width = upper - lower
            
            prediction_intervals[conf_level] = {
                'lower': lower,
                'upper': upper,
                'width': interval_width
            }
        
        # Create subplots with cartopy projection if available
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': projection})
        axes = axes.flatten()
        
        # Plot 1: Ensemble mean
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[0])
            cmap_settings = self._get_colormap_settings('variable')
            im1 = axes[0].pcolormesh(lon_2d, lat_2d, ensemble_mean.values, 
                                    transform=ccrs.PlateCarree(), **cmap_settings)
        else:
            cmap_settings = self._get_colormap_settings('variable')
            im1 = ensemble_mean.plot(ax=axes[0], add_colorbar=False, **cmap_settings)
        
        axes[0].set_title(f'{var_name} - Ensemble Mean')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Plot 2: Ensemble standard deviation (uncertainty)
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[1])
            std_cmap_settings = self._get_colormap_settings('std')
            im2 = axes[1].pcolormesh(lon_2d, lat_2d, ensemble_std.values, 
                                    transform=ccrs.PlateCarree(), **std_cmap_settings)
        else:
            std_cmap_settings = self._get_colormap_settings('std')
            im2 = ensemble_std.plot(ax=axes[1], add_colorbar=False, **std_cmap_settings)
        
        axes[1].set_title(f'{var_name} - Prediction Uncertainty (Std)')
        plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Standard Deviation')
        
        # Plot 3: Prediction interval width (default: 95% confidence)
        conf_level_display = 0.95 if 0.95 in confidence_levels else confidence_levels[-1]
        interval_width = prediction_intervals[conf_level_display]['width']
        
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[2])
            width_cmap_settings = self._get_colormap_settings('metric')
            im3 = axes[2].pcolormesh(lon_2d, lat_2d, interval_width.values, 
                                    transform=ccrs.PlateCarree(), **width_cmap_settings)
        else:
            width_cmap_settings = self._get_colormap_settings('metric')
            im3 = interval_width.plot(ax=axes[2], add_colorbar=False, **width_cmap_settings)
        
        axes[2].set_title(f'{var_name} - {conf_level_display*100:.0f}% Prediction Interval Width')
        plt.colorbar(im3, ax=axes[2], shrink=0.8, label='Interval Width')
        
        # Plot 4: Coefficient of variation (relative uncertainty)
        # Avoid division by zero
        ensemble_mean_safe = np.where(np.abs(ensemble_mean) > 1e-10, ensemble_mean, np.nan)
        coeff_variation = ensemble_std / np.abs(ensemble_mean_safe)
        
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[3])
            cv_cmap_settings = self._get_colormap_settings('metric')
            im4 = axes[3].pcolormesh(lon_2d, lat_2d, coeff_variation.values, 
                                    transform=ccrs.PlateCarree(), **cv_cmap_settings)
        else:
            cv_cmap_settings = self._get_colormap_settings('metric')
            im4 = coeff_variation.plot(ax=axes[3], add_colorbar=False, **cv_cmap_settings)
        
        axes[3].set_title(f'{var_name} - Coefficient of Variation')
        plt.colorbar(im4, ax=axes[3], shrink=0.8, label='CV = Std/|Mean|')
        
        # Add summary statistics as text
        mean_uncertainty = float(np.nanmean(ensemble_std.values))
        max_uncertainty = float(np.nanmax(ensemble_std.values))
        mean_cv = float(np.nanmean(coeff_variation.values))
        
        stats_text = (f'Mean Uncertainty: {mean_uncertainty:.4f}\n'
                     f'Max Uncertainty: {max_uncertainty:.4f}\n'
                     f'Mean CV: {mean_cv:.4f}')
        
        fig.text(0.02, 0.95, stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Uncertainty Quantification (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"uncertainty_quantification_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ Uncertainty quantification plot saved: {os.path.basename(output_path)}')
        
        return output_path