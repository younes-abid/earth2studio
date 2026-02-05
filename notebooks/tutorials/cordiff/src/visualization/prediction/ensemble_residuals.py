# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Ensemble residuals visualization - Residuals between ensemble members and ground truth.
Corresponds to config: PLOT_ENSEMBLE_RESIDUALS = True
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base import BasePlot


class EnsembleResidualsPlot(BasePlot):
    """
    Creates ensemble residuals plots comparing predictions to ground truth.
    """
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate ensemble residuals plots.
        
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
        print(f'🎨 Creating ensemble residuals plots for time index {time_idx}...')
        
        # Try to load ground truth data
        try:
            with xr.open_dataset(netcdf_path, group='truth') as truth_ds:
                var_name = self.config.OUTPUT_VARIABLES[variable_idx]
                if var_name not in truth_ds.data_vars:
                    print(f'   ⚠️  Ground truth for {var_name} not available, skipping residuals')
                    return None
                ground_truth = truth_ds[var_name].isel(time=time_idx)
        except (OSError, KeyError):
            print(f'   ⚠️  Ground truth data not available, skipping residuals')
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
        
        # Calculate grid layout for ensemble members
        n_members = self.config.NUM_ENSEMBLES
        n_rows, n_cols = self._create_grid_subplot(n_members, max_cols=4)
        
        # Create subplots with cartopy projection if available
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), 
                                subplot_kw={'projection': projection})
        axes = axes.flatten() if n_members > 1 else [axes]
        
        # Get colormap settings for residuals (diverging colormap)
        cmap_settings = self._get_colormap_settings('residual')
        
        # Calculate symmetric color limits for residuals
        residuals_values = residuals.values
        vmax = np.nanpercentile(np.abs(residuals_values), 95)
        cmap_settings.update({'vmin': -vmax, 'vmax': vmax})
        
        # Plot residuals for each ensemble member
        for i in range(n_members):
            member_residuals = residuals.isel(ensemble=i)
            
            if self.config.USE_CARTOPY:
                self._setup_geographic_axes(axes[i])
                im = axes[i].pcolormesh(lon_2d, lat_2d, member_residuals.values, 
                                       transform=ccrs.PlateCarree(), **cmap_settings)
            else:
                im = member_residuals.plot(ax=axes[i], add_colorbar=False, **cmap_settings)
            
            # Calculate member statistics
            rmse = float(np.sqrt(np.nanmean(member_residuals.values**2)))
            bias = float(np.nanmean(member_residuals.values))
            
            axes[i].set_title(f'Member {i+1} Residuals\nRMSE: {rmse:.3f}, Bias: {bias:.3f}')
            plt.colorbar(im, ax=axes[i], shrink=0.8, label='Residual')
        
        # Hide extra subplots
        for i in range(n_members, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{var_name} - Ensemble Residuals vs Ground Truth (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"ensemble_residuals_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ Ensemble residuals plot saved: {os.path.basename(output_path)}')
        
        return output_path