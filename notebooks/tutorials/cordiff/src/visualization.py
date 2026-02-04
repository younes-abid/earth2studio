# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Visualization functions for CorrDiff ensemble results.
Minimal, efficient plotting using xarray built-in capabilities.
"""

import os
import matplotlib.pyplot as plt
import xarray as xr
from config import EnsembleConfig


def plot_ensemble_members_grid(config: EnsembleConfig, netcdf_path: str, time_idx: int = 0) -> None:
    """
    Plot all ensemble members in a grid layout.
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object
    netcdf_path : str
        Path to NetCDF ensemble file
    time_idx : int
        Time index to plot
    """
    print(f'🎨 Creating ensemble members grid for time index {time_idx}...')
    
    with xr.open_dataset(netcdf_path, group='prediction') as ds:
        var_name = config.OUTPUT_VARIABLES[0]
        data = ds[var_name].isel(time=time_idx)  # Shape: (ensemble, y, x)
        
        # Calculate grid layout
        n_members = config.NUM_ENSEMBLES
        cols = int(n_members**0.5) + (1 if n_members**0.5 % 1 else 0)
        rows = (n_members + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = axes.flatten() if n_members > 1 else [axes]
        
        # Plot each ensemble member using xarray's plot
        for i in range(n_members):
            member_data = data.isel(ensemble=i)
            im = member_data.plot(ax=axes[i], cmap='viridis', add_colorbar=False)
            axes[i].set_title(f'Member {i+1}')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        # Hide extra subplots
        for i in range(n_members, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{var_name} - All Ensemble Members (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot with time_idx folder structure and direct timestamp usage
        output_path = f"{config.OUTPUT_ANALYSIS_FOLDER}/{time_idx}/ensemble_members_{config.VARIABLES}_{config.TIMESTAMP}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'   ✅ Ensemble members grid saved: {output_path}')
        plt.show()


def plot_ensemble_statistics(config: EnsembleConfig, netcdf_path: str, time_idx: int = 0) -> None:
    """
    Plot ensemble mean, max, min, and standard deviation.
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object
    netcdf_path : str
        Path to NetCDF ensemble file  
    time_idx : int
        Time index to plot
    """
    print(f'🎨 Creating ensemble statistics plots for time index {time_idx}...')
    
    with xr.open_dataset(netcdf_path, group='prediction') as ds:
        var_name = config.OUTPUT_VARIABLES[0]
        data = ds[var_name].isel(time=time_idx)  # Shape: (ensemble, y, x)
        
        # Compute statistics using xarray
        stats = {
            'Mean': data.mean(dim='ensemble'),
            'Std Dev': data.std(dim='ensemble'),
            'Min': data.min(dim='ensemble'),
            'Max': data.max(dim='ensemble')
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot each statistic using xarray's plot
        for i, (stat_name, stat_data) in enumerate(stats.items()):
            cmap = 'viridis' if stat_name in ['Mean', 'Min', 'Max'] else 'Reds'
            im = stat_data.plot(ax=axes[i], cmap=cmap, add_colorbar=False)
            axes[i].set_title(f'{var_name} - {stat_name}')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        plt.suptitle(f'{var_name} - Ensemble Statistics (Time {time_idx})')
        plt.tight_layout()
        
        # Save plot with time_idx folder structure and direct timestamp usage
        output_path = f"{config.OUTPUT_ANALYSIS_FOLDER}/{time_idx}/ensemble_statistics_{config.VARIABLES}_{config.TIMESTAMP}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'   ✅ Ensemble statistics plot saved: {output_path}')
        plt.show()


def print_ensemble_summary(config, results, saved_file, ensemble_stats):
    """
    Print a comprehensive summary of the ensemble workflow results.
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object
    results : dict
        Results from inference
    saved_file : str
        Path to saved ensemble file
    ensemble_stats : dict
        Ensemble statistics
    """
    print('🎉 ENSEMBLE CORRDIFF INFERENCE COMPLETE!')
    print('=' * 60)
    print(f'📊 Configuration Summary:')
    print(f'   • Ensemble members: {config.NUM_ENSEMBLES}')
    print(f'   • Sampling mode: {config.SAMPLING_MODE}')
    print(f'   • Diffusion steps: {config.NUMBER_OF_STEPS}')
    print(f'   • Time steps processed: {len(results["times"])}')
    print(f'   • Variables: {config.OUTPUT_VARIABLES}')
    
    print(f'💾 Output Files:')
    print(f'   • Ensemble data: {saved_file}')
    print(f'   • Analysis folder: {config.OUTPUT_ANALYSIS_FOLDER}')
    
    print(f'📈 Key Insights:')
    for var_name in config.OUTPUT_VARIABLES:
        if var_name in ensemble_stats:
            stats = ensemble_stats[var_name]
            uncertainty = float(stats['std'].mean())
            spread = float(stats['range'].mean())
            print(f'   • {var_name}: Average uncertainty = {uncertainty:.4f}, Average spread = {spread:.4f}')
    
    print(f'\n🔄 To load and analyze the results:')
    print(f'```python')
    print(f'import xarray as xr')
    print(f'pred_ds = xr.open_dataset("{saved_file}", group="prediction")')
    print(f'ensemble_data = pred_ds["{config.OUTPUT_VARIABLES[0]}"]')
    print(f'```')
    
    print(f'\n🚀 Next Steps:')
    print(f'   1. Experiment with different SAMPLING_MODE settings')
    print(f'   2. Adjust NUM_ENSEMBLES for uncertainty requirements')
    print(f'   3. Check analysis folder for saved plots and statistics')
    print('=' * 60)


# Main plotting function for backward compatibility
def plot_ensemble_analysis(config, netcdf_path, time_idx=0):
    """
    Generate both ensemble plots (members grid + statistics).
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object
    netcdf_path : str
        Path to NetCDF ensemble file
    time_idx : int
        Time index to plot
    """
    plot_ensemble_members_grid(config, netcdf_path, time_idx)
    plot_ensemble_statistics(config, netcdf_path, time_idx)