# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Ensemble members visualization - Grid plot of all ensemble members.
Corresponds to config: PLOT_ENSEMBLE_MEMBERS = True
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from ..base import BasePlot


class EnsembleMembersPlot(BasePlot):
    """
    Creates grid plot of all ensemble members.
    """
    
    def plot(self, netcdf_path, time_idx=0, variable_idx=0, **kwargs):
        """
        Generate grid plot of ensemble members.
        
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
        print(f'🎨 Creating ensemble members grid for time index {time_idx}...')
        
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
        
        # Calculate grid layout
        n_members = self.config.NUM_ENSEMBLES
        n_rows, n_cols = self._create_grid_subplot(n_members, max_cols=4)
        
        # Create subplots with cartopy projection if available
        projection = ccrs.PlateCarree() if self.config.USE_CARTOPY else None
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), 
                                subplot_kw={'projection': projection})
        axes = axes.flatten() if n_members > 1 else [axes]
        
        # Get colormap settings
        cmap_settings = self._get_colormap_settings('variable')
        
        # Plot each ensemble member
        for i in range(n_members):
            member_data = data.isel(ensemble=i)
            
            if self.config.USE_CARTOPY:
                self._setup_geographic_axes(axes[i])
                im = axes[i].pcolormesh(lon_2d, lat_2d, member_data.values, 
                                       transform=ccrs.PlateCarree(), **cmap_settings)
            else:
                im = member_data.plot(ax=axes[i], add_colorbar=False, **cmap_settings)
            
            axes[i].set_title(f'Member {i+1}')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        # Hide extra subplots
        for i in range(n_members, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{var_name} - All Ensemble Members (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot
        filename = f"ensemble_members_{var_name}_time{time_idx:03d}_{self.config.TIMESTAMP}"
        output_path = self._save_plot(fig, filename)
        print(f'   ✅ Ensemble members grid saved: {os.path.basename(output_path)}')
        
        return output_path