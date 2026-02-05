# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Residuals statistics visualization - Statistical analysis of ensemble residuals.
Corresponds to config: PLOT_ENSEMBLE_RESIDUALS_STATISTICS = True
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base import BasePlot


class ResidualsStatisticsPlot(BasePlot):
    """
    Creates statistical analysis plots of ensemble residuals.
    """
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate residuals statistics plots.
        
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
        print(f'🎨 Creating residuals statistics plots for time index {time_idx}...')
        
        # Try to load ground truth data
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                var_name = self.config.OUTPUT_VARIABLES[variable_idx]
                if var_name not in truth_ds.data_vars:
                    print(f'   ⚠️  Ground truth for {var_name} not available, skipping residuals statistics')
                    return None
                ground_truth = truth_ds[var_name].isel(time=time_idx)
        except (OSError, KeyError):
            print(f'   ⚠️  Ground truth data not available, skipping residuals statistics')
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
        
        # Calculate residuals for each ensemble member
        residuals = predictions - ground_truth  # Broadcasting: (ensemble, y, x) - (y, x)
        
        # Compute residuals statistics across ensemble members
        residuals_stats = {
            'Mean Residuals': residuals.mean(dim='ensemble'),
            'Std of Residuals': residuals.std(dim='ensemble'),
            'RMSE': np.sqrt((residuals**2).mean(dim='ensemble')),
            'Abs Mean Residuals': np.abs(residuals).mean(dim='ensemble')
        }
        
        # Create subplots with cartopy projection if available
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': projection})
        axes = axes.flatten()
        
        # Plot each residuals statistic
        for i, (stat_name, stat_data) in enumerate(residuals_stats.items()):
            if self.config.USE_CARTOPY:
                self._setup_geographic_axes(axes[i])
                
                # Choose colormap based on statistic type
                if 'Mean' in stat_name and 'Abs' not in stat_name:
                    cmap_settings = self._get_colormap_settings('residual')
                    # Symmetric limits for mean residuals
                    vmax = np.nanpercentile(np.abs(stat_data.values), 95)
                    cmap_settings.update({'vmin': -vmax, 'vmax': vmax})
                else:
                    cmap_settings = self._get_colormap_settings('metric')
                
                im = axes[i].pcolormesh(lon_2d, lat_2d, stat_data.values, 
                                       transform=ccrs.PlateCarree(), **cmap_settings)
            else:
                # Choose colormap based on statistic type
                if 'Mean' in stat_name and 'Abs' not in stat_name:
                    cmap_settings = self._get_colormap_settings('residual')
                    # Symmetric limits for mean residuals
                    vmax = np.nanpercentile(np.abs(stat_data.values), 95)
                    cmap_settings.update({'vmin': -vmax, 'vmax': vmax})
                else:
                    cmap_settings = self._get_colormap_settings('metric')
                    
                im = stat_data.plot(ax=axes[i], add_colorbar=False, **cmap_settings)
            
            axes[i].set_title(f'{var_name} - {stat_name}')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
            
            # Add global statistics as text
            global_mean = float(np.nanmean(stat_data.values))
            global_std = float(np.nanstd(stat_data.values))
            axes[i].text(0.02, 0.98, f'μ={global_mean:.4f}\nσ={global_std:.4f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Ensemble Residuals Statistics (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"residuals_statistics_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ Residuals statistics plot saved: {os.path.basename(output_path)}')
        
        return output_path