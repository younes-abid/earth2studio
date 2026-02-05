# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Truth vs Mean visualization - Side-by-side comparison of ground truth and ensemble mean.
Corresponds to config: PLOT_GROUND_TRUTH_VS_ENSEMBLEMEAN = True
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base import BasePlot


class TruthVsMeanPlot(BasePlot):
    """
    Creates side-by-side comparison of ground truth and ensemble mean.
    """
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate truth vs ensemble mean comparison plots.
        
        Parameters
        ----------
        netcdf_path : str
            Path to NetCDF ensemble file
        time_idx : int
            Time index to plot
        variable_idx : int
            Variable index to plot (if multiple variables)
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        str or None
            Path to saved plot file, or None if ground truth not available
        """
        print(f'🎨 Creating truth vs ensemble mean comparison for time index {time_idx}...')
        
        # Try to load ground truth data
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                var_name = self.config.OUTPUT_VARIABLES[variable_idx]
                if var_name not in truth_ds.data_vars:
                    print(f'   ⚠️  Ground truth for {var_name} not available, skipping comparison')
                    return None
                ground_truth = truth_ds[var_name].isel(time=time_idx)
        except (OSError, KeyError):
            print(f'   ⚠️  Ground truth data not available, skipping comparison')
            return None
        
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
        
        # Calculate ensemble mean
        ensemble_mean = predictions.mean(dim='ensemble')
        
        # Calculate difference
        difference = ensemble_mean - ground_truth
        
        # Create subplots with cartopy projection if available
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': projection})
        
        # Get colormap settings
        cmap_settings = self._get_colormap_settings('variable')
        
        # Plot ground truth
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[0])
            im1 = axes[0].pcolormesh(lon_2d, lat_2d, ground_truth.values, 
                                    transform=ccrs.PlateCarree(), **cmap_settings)
        else:
            im1 = ground_truth.plot(ax=axes[0], add_colorbar=False, **cmap_settings)
        
        axes[0].set_title(f'{var_name} - Ground Truth')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Plot ensemble mean
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[1])
            im2 = axes[1].pcolormesh(lon_2d, lat_2d, ensemble_mean.values, 
                                    transform=ccrs.PlateCarree(), **cmap_settings)
        else:
            im2 = ensemble_mean.plot(ax=axes[1], add_colorbar=False, **cmap_settings)
        
        axes[1].set_title(f'{var_name} - Ensemble Mean')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Plot difference with diverging colormap
        diff_cmap_settings = self._get_colormap_settings('residual')
        # Symmetric limits for difference
        vmax_diff = np.nanpercentile(np.abs(difference.values), 95)
        diff_cmap_settings.update({'vmin': -vmax_diff, 'vmax': vmax_diff})
        
        if self.config.USE_CARTOPY:
            self._setup_geographic_axes(axes[2])
            im3 = axes[2].pcolormesh(lon_2d, lat_2d, difference.values, 
                                    transform=ccrs.PlateCarree(), **diff_cmap_settings)
        else:
            im3 = difference.plot(ax=axes[2], add_colorbar=False, **diff_cmap_settings)
        
        axes[2].set_title(f'{var_name} - Difference (Mean - Truth)')
        plt.colorbar(im3, ax=axes[2], shrink=0.8, label='Difference')
        
        # Calculate and display metrics
        rmse = float(np.sqrt(np.nanmean(difference.values**2)))
        mae = float(np.nanmean(np.abs(difference.values)))
        bias = float(np.nanmean(difference.values))
        corr = float(np.corrcoef(ground_truth.values.flatten(), 
                                ensemble_mean.values.flatten())[0, 1])
        
        # Add metrics as text
        metrics_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nBias: {bias:.4f}\nCorr: {corr:.4f}'
        fig.text(0.02, 0.95, metrics_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Ground Truth vs Ensemble Mean Comparison (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"truth_vs_mean_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ Truth vs mean comparison plot saved: {os.path.basename(output_path)}')
        
        return output_path