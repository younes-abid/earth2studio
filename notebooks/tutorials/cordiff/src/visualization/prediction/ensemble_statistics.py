# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Ensemble statistics visualization - Mean, max, min, and standard deviation plots.
Corresponds to config: PLOT_ENSEMBLE_STATISTICS = True
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base import BasePlot


class EnsembleStatisticsPlot(BasePlot):
    """
    Creates ensemble statistics plots (mean, std, min, max).
    """
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate ensemble statistics plots.
        
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
        str
            Path to saved plot file
        """
        print(f'🎨 Creating ensemble statistics plots for time index {time_idx}...')
        
        # Load data
        with xr.open_dataset(netcdf_path, group='prediction') as ds:
            var_name = self.config.OUTPUT_VARIABLES[variable_idx]
            data = ds[var_name].isel(time=time_idx)  # Shape: (ensemble, y, x)
            
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
        
        # Compute statistics using xarray
        stats = {
            'Mean': data.mean(dim='ensemble'),
            'Std Dev': data.std(dim='ensemble'),
            'Min': data.min(dim='ensemble'),
            'Max': data.max(dim='ensemble')
        }
        
        # Create subplots with cartopy projection if available
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': projection})
        axes = axes.flatten()
        
        # Plot each statistic
        for i, (stat_name, stat_data) in enumerate(stats.items()):
            if self.config.USE_CARTOPY:
                self._setup_geographic_axes(axes[i])
                
                # Choose colormap based on statistic type
                if stat_name == 'Std Dev':
                    cmap_settings = self._get_colormap_settings('std')
                else:
                    cmap_settings = self._get_colormap_settings('variable')
                
                im = axes[i].pcolormesh(lon_2d, lat_2d, stat_data.values, 
                                       transform=ccrs.PlateCarree(), **cmap_settings)
            else:
                # Choose colormap based on statistic type
                if stat_name == 'Std Dev':
                    cmap_settings = self._get_colormap_settings('std')
                else:
                    cmap_settings = self._get_colormap_settings('variable')
                    
                im = stat_data.plot(ax=axes[i], add_colorbar=False, **cmap_settings)
            
            axes[i].set_title(f'{var_name} - {stat_name}')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
            
            # Add statistics as text
            mean_val = float(stat_data.mean())
            std_val = float(stat_data.std())
            axes[i].text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'{var_name} - Ensemble Statistics (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"ensemble_statistics_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ Ensemble statistics plot saved: {os.path.basename(output_path)}')
        
        return output_path