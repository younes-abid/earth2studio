"""
Visualization functions for CorrDiff ensemble results.
"""
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Cartopy not available. Geographic projections will not be used.")


def setup_matplotlib_style():
    """Setup matplotlib style for better plots."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.dpi': 100,
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })


def detect_coordinate_system(dataset: xr.Dataset, verbose: bool = True) -> Dict[str, Any]:
    """
    Auto-detect coordinate system and grid information.
    
    Args:
        dataset: xarray Dataset
        verbose: Enable verbose output
        
    Returns:
        Dictionary with coordinate system information
    """
    coord_info = {
        'has_geographic_coords': False,
        'lat_name': None,
        'lon_name': None,
        'x_name': None,
        'y_name': None,
        'use_cartopy': False,
        'extent': None
    }
    
    # Look for coordinate variables
    coord_names = list(dataset.coords.keys())
    
    # Check for geographic coordinates
    for name in coord_names:
        if name.lower() in ['lat', 'latitude', 'y']:
            coord_info['lat_name'] = name
        if name.lower() in ['lon', 'longitude', 'x']:
            coord_info['lon_name'] = name
    
    # Check for grid coordinates
    if 'x' in coord_names:
        coord_info['x_name'] = 'x'
    if 'y' in coord_names:
        coord_info['y_name'] = 'y'
    
    # Determine if we have geographic coordinates
    if coord_info['lat_name'] and coord_info['lon_name']:
        lat_data = dataset[coord_info['lat_name']]
        lon_data = dataset[coord_info['lon_name']]
        
        # Check if coordinates look geographic
        if (-90 <= lat_data.min() <= 90 and -90 <= lat_data.max() <= 90 and
            -180 <= lon_data.min() <= 360 and -180 <= lon_data.max() <= 360):
            coord_info['has_geographic_coords'] = True
            coord_info['use_cartopy'] = HAS_CARTOPY
            coord_info['extent'] = [lon_data.min(), lon_data.max(), 
                                  lat_data.min(), lat_data.max()]
    
    if verbose:
        if coord_info['has_geographic_coords']:
            print(f"✅ Detected geographic coordinates: {coord_info['lat_name']}, {coord_info['lon_name']}")
            if coord_info['use_cartopy']:
                print("🗺️  Will use Cartopy for geographic projections")
            else:
                print("📍 Will use simple plotting (no Cartopy)")
        else:
            print("⚠️  No geographic coordinates detected, using index-based plotting")
    
    return coord_info


def create_ensemble_member_plots(pred_dataset: xr.Dataset, variables: List[str], 
                                time_idx: int = 0, output_dir: str = None,
                                coord_info: Dict = None, verbose: bool = True) -> List[str]:
    """
    Create plots showing individual ensemble members.
    
    Args:
        pred_dataset: Prediction dataset with ensemble dimension
        variables: List of variables to plot
        time_idx: Time index to plot
        output_dir: Directory to save plots
        coord_info: Coordinate system information
        verbose: Enable verbose output
        
    Returns:
        List of saved plot filenames
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print('🎨 Creating ensemble member plots...')
    
    saved_files = []
    
    for var_name in variables:
        if var_name not in pred_dataset:
            continue
        
        var_data = pred_dataset[var_name]  # (ensemble, time, y, x)
        if 'time' in var_data.dims:
            var_data = var_data.isel(time=time_idx)
        
        n_ensemble = var_data.sizes['ensemble']
        
        # Determine grid layout
        n_cols = min(4, n_ensemble)
        n_rows = (n_ensemble + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_ensemble == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'{var_name} - All Ensemble Members (Time: {time_idx})', 
                    fontsize=14, fontweight='bold')
        
        # Plot each ensemble member
        vmin, vmax = var_data.min().values, var_data.max().values
        
        for i in range(n_ensemble):
            ax = axes[i]
            data_slice = var_data.isel(ensemble=i)
            
            if coord_info and coord_info['use_cartopy']:
                # Use cartopy projection
                ax.remove()
                ax = fig.add_subplot(n_rows, n_cols, i+1, 
                                   projection=ccrs.PlateCarree())
                
                im = ax.pcolormesh(pred_dataset[coord_info['lon_name']], 
                                 pred_dataset[coord_info['lat_name']], 
                                 data_slice, 
                                 transform=ccrs.PlateCarree(),
                                 vmin=vmin, vmax=vmax, cmap='viridis')
                ax.coastlines()
                ax.gridlines(alpha=0.5)
                ax.add_feature(cfeature.BORDERS, alpha=0.5)
            else:
                # Simple matplotlib imshow
                im = ax.imshow(data_slice, cmap='viridis', vmin=vmin, vmax=vmax, 
                              origin='lower', aspect='auto')
            
            ax.set_title(f'Member {i+1}')
        
        # Hide extra subplots
        for i in range(n_ensemble, len(axes)):
            axes[i].set_visible(False)
        
        # Add colorbar
        plt.tight_layout()
        cbar = plt.colorbar(im, ax=axes[:n_ensemble], shrink=0.6, aspect=30)
        cbar.set_label(var_name)
        
        if output_dir:
            filename = f'{var_name}_ensemble_members_t{time_idx}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            saved_files.append(filepath)
            if verbose:
                print(f'   Saved: {filename}')
        
        plt.show()
        plt.close()
    
    if verbose:
        elapsed = time.time() - start_time
        print(f'✅ Ensemble member plots created in {elapsed:.3f}s')
    
    return saved_files


def create_ensemble_statistics_plots(stats: Dict, variables: List[str], 
                                   time_idx: int = 0, output_dir: str = None,
                                   coord_info: Dict = None, verbose: bool = True) -> List[str]:
    """
    Create plots showing ensemble statistics (mean, std, etc.).
    
    Args:
        stats: Dictionary of ensemble statistics
        variables: List of variables to plot
        time_idx: Time index to plot
        output_dir: Directory to save plots
        coord_info: Coordinate system information
        verbose: Enable verbose output
        
    Returns:
        List of saved plot filenames
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print('📊 Creating ensemble statistics plots...')
    
    saved_files = []
    
    for var_name in variables:
        if var_name not in stats:
            continue
        
        var_stats = stats[var_name]
        
        # Create 2x3 subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{var_name} - Ensemble Statistics (Time: {time_idx})', 
                    fontsize=16, fontweight='bold')
        
        stats_to_plot = [
            ('mean', 'Ensemble Mean', 'viridis'),
            ('std', 'Ensemble Std Dev', 'Reds'),
            ('min', 'Ensemble Min', 'Blues'),
            ('max', 'Ensemble Max', 'Oranges'),
            ('range', 'Ensemble Range', 'plasma'),
            ('iqr', 'Interquartile Range', 'RdYlBu_r')
        ]
        
        for idx, (stat_name, title, cmap) in enumerate(stats_to_plot):
            if stat_name not in var_stats:
                continue
                
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            data = var_stats[stat_name]
            if 'time' in data.dims:
                data = data.isel(time=time_idx)
            
            if coord_info and coord_info['use_cartopy']:
                # Use cartopy projection
                ax.remove()
                ax = fig.add_subplot(2, 3, idx+1, projection=ccrs.PlateCarree())
                
                im = ax.pcolormesh(var_stats[coord_info['lon_name']], 
                                 var_stats[coord_info['lat_name']], 
                                 data, 
                                 transform=ccrs.PlateCarree(),
                                 cmap=cmap)
                ax.coastlines()
                ax.gridlines(alpha=0.5)
                ax.add_feature(cfeature.BORDERS, alpha=0.5)
            else:
                # Simple matplotlib imshow
                im = ax.imshow(data, cmap=cmap, origin='lower', aspect='auto')
            
            ax.set_title(title)
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        plt.tight_layout()
        
        if output_dir:
            filename = f'{var_name}_ensemble_statistics_t{time_idx}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            saved_files.append(filepath)
            if verbose:
                print(f'   Saved: {filename}')
        
        plt.show()
        plt.close()
    
    if verbose:
        elapsed = time.time() - start_time
        print(f'✅ Ensemble statistics plots created in {elapsed:.3f}s')
    
    return saved_files


def create_metrics_plots(metrics: Dict, variables: List[str], 
                        output_dir: str = None, verbose: bool = True) -> List[str]:
    """
    Create plots showing performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
        variables: List of variables to plot
        output_dir: Directory to save plots
        verbose: Enable verbose output
        
    Returns:
        List of saved plot filenames
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print('📏 Creating performance metrics plots...')
    
    saved_files = []
    
    for var_name in variables:
        if var_name not in metrics:
            continue
        
        var_metrics = metrics[var_name]
        
        # Create figure for metrics
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{var_name} - Performance Metrics', 
                    fontsize=16, fontweight='bold')
        
        # Time series metrics if available
        metric_names = ['rmse', 'mae', 'correlation', 'r2', 'bias', 'reliability']
        
        for idx, metric_name in enumerate(metric_names):
            if metric_name not in var_metrics:
                continue
                
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            metric_data = var_metrics[metric_name]
            
            if 'time' in metric_data.dims and len(metric_data.time) > 1:
                # Plot time series
                ax.plot(range(len(metric_data.time)), metric_data, 'o-', linewidth=2)
                ax.set_xlabel('Time Step')
                ax.set_ylabel(metric_name.upper())
                ax.set_title(f'{metric_name.upper()} over Time')
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = metric_data.mean().values
                std_val = metric_data.std().values
                ax.text(0.05, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                # Single value - show as bar
                metric_val = float(metric_data.values) if hasattr(metric_data, 'values') else metric_data
                ax.bar([metric_name], [metric_val])
                ax.set_title(f'{metric_name.upper()}: {metric_val:.4f}')
                ax.set_ylabel(metric_name.upper())
        
        # Hide empty subplots
        for idx in range(len(metric_names), 6):
            row = idx // 3
            col = idx % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if output_dir:
            filename = f'{var_name}_metrics.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            saved_files.append(filepath)
            if verbose:
                print(f'   Saved: {filename}')
        
        plt.show()
        plt.close()
    
    if verbose:
        elapsed = time.time() - start_time
        print(f'✅ Metrics plots created in {elapsed:.3f}s')
    
    return saved_files


def create_histogram_plots(pred_dataset: xr.Dataset, truth_dataset: xr.Dataset = None,
                          variables: List[str] = None, output_dir: str = None,
                          verbose: bool = True) -> List[str]:
    """
    Create histogram analysis plots.
    
    Args:
        pred_dataset: Prediction dataset
        truth_dataset: Ground truth dataset (optional)
        variables: List of variables to plot
        output_dir: Directory to save plots
        verbose: Enable verbose output
        
    Returns:
        List of saved plot filenames
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print('📈 Creating histogram analysis plots...')
    
    saved_files = []
    
    for var_name in variables:
        if var_name not in pred_dataset:
            continue
        
        pred_var = pred_dataset[var_name]  # (ensemble, time, y, x)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{var_name} - Distribution Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. Ensemble member distributions
        ax = axes[0, 0]
        if 'ensemble' in pred_var.dims:
            for i in range(min(5, pred_var.sizes['ensemble'])):  # Show max 5 members
                data_flat = pred_var.isel(ensemble=i).values.flatten()
                ax.hist(data_flat, bins=50, alpha=0.6, density=True, 
                       label=f'Member {i+1}')
        else:
            data_flat = pred_var.values.flatten()
            ax.hist(data_flat, bins=50, alpha=0.7, density=True, label='Prediction')
        
        ax.set_xlabel(var_name)
        ax.set_ylabel('Density')
        ax.set_title('Ensemble Member Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Ensemble statistics distributions
        ax = axes[0, 1]
        if 'ensemble' in pred_var.dims:
            ens_mean = pred_var.mean(dim='ensemble')
            ens_std = pred_var.std(dim='ensemble')
            
            ax.hist(ens_mean.values.flatten(), bins=50, alpha=0.7, density=True, 
                   label='Ensemble Mean', color='blue')
            
            # Add truth if available
            if truth_dataset and var_name in truth_dataset:
                truth_var = truth_dataset[var_name]
                ax.hist(truth_var.values.flatten(), bins=50, alpha=0.7, density=True, 
                       label='Truth', color='red')
        
        ax.set_xlabel(var_name)
        ax.set_ylabel('Density')
        ax.set_title('Mean vs Truth Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Ensemble spread distribution
        ax = axes[1, 0]
        if 'ensemble' in pred_var.dims:
            ens_std = pred_var.std(dim='ensemble')
            ax.hist(ens_std.values.flatten(), bins=50, alpha=0.7, density=True, 
                   color='green')
            ax.set_xlabel('Standard Deviation')
            ax.set_ylabel('Density')
            ax.set_title('Ensemble Spread Distribution')
            ax.grid(True, alpha=0.3)
        
        # 4. QQ plot or error distribution
        ax = axes[1, 1]
        if truth_dataset and var_name in truth_dataset and 'ensemble' in pred_var.dims:
            truth_var = truth_dataset[var_name]
            ens_mean = pred_var.mean(dim='ensemble')
            errors = (ens_mean - truth_var).values.flatten()
            
            ax.hist(errors, bins=50, alpha=0.7, density=True, color='orange')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Density')
            ax.set_title('Prediction Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # Just show ensemble variance if no truth
            if 'ensemble' in pred_var.dims:
                ens_var = pred_var.var(dim='ensemble')
                ax.hist(ens_var.values.flatten(), bins=50, alpha=0.7, density=True, 
                       color='purple')
                ax.set_xlabel('Variance')
                ax.set_ylabel('Density')
                ax.set_title('Ensemble Variance Distribution')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            filename = f'{var_name}_histograms.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            saved_files.append(filepath)
            if verbose:
                print(f'   Saved: {filename}')
        
        plt.show()
        plt.close()
    
    if verbose:
        elapsed = time.time() - start_time
        print(f'✅ Histogram plots created in {elapsed:.3f}s')
    
    return saved_files


def create_comprehensive_visualization(pred_dataset: xr.Dataset, 
                                     truth_dataset: xr.Dataset = None,
                                     stats: Dict = None, metrics: Dict = None,
                                     variables: List[str] = None, 
                                     time_idx: int = 0,
                                     output_dir: str = None,
                                     verbose: bool = True) -> Dict[str, List[str]]:
    """
    Create comprehensive visualization suite.
    
    Args:
        pred_dataset: Prediction dataset
        truth_dataset: Ground truth dataset (optional)
        stats: Ensemble statistics dictionary (optional)
        metrics: Performance metrics dictionary (optional)
        variables: List of variables to visualize
        time_idx: Time index for spatial plots
        output_dir: Directory to save plots
        verbose: Enable verbose output
        
    Returns:
        Dictionary with lists of saved files by plot type
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print('🎨 Creating comprehensive visualization suite...')
    
    setup_matplotlib_style()
    
    # Auto-detect coordinate system
    coord_info = detect_coordinate_system(pred_dataset, verbose)
    
    saved_files = {
        'ensemble_members': [],
        'ensemble_stats': [],
        'metrics': [],
        'histograms': []
    }
    
    if variables is None:
        variables = [var for var in pred_dataset.data_vars.keys() 
                    if 'ensemble' in pred_dataset[var].dims]
    
    # Create ensemble member plots
    saved_files['ensemble_members'] = create_ensemble_member_plots(
        pred_dataset, variables, time_idx, output_dir, coord_info, verbose
    )
    
    # Create ensemble statistics plots
    if stats:
        saved_files['ensemble_stats'] = create_ensemble_statistics_plots(
            stats, variables, time_idx, output_dir, coord_info, verbose
        )
    
    # Create metrics plots
    if metrics:
        saved_files['metrics'] = create_metrics_plots(
            metrics, variables, output_dir, verbose
        )
    
    # Create histogram plots
    saved_files['histograms'] = create_histogram_plots(
        pred_dataset, truth_dataset, variables, output_dir, verbose
    )
    
    if verbose:
        elapsed = time.time() - start_time
        total_plots = sum(len(files) for files in saved_files.values())
        print(f'✅ Comprehensive visualization completed in {elapsed:.3f}s')
        print(f'📊 Total plots created: {total_plots}')
        
        if output_dir:
            print(f'💾 All plots saved to: {output_dir}')
    
    return saved_files


def create_summary_report(pred_dataset: xr.Dataset, truth_dataset: xr.Dataset = None,
                         stats: Dict = None, metrics: Dict = None,
                         config: Dict = None, output_dir: str = None,
                         verbose: bool = True) -> str:
    """
    Create a summary report figure.
    
    Args:
        pred_dataset: Prediction dataset
        truth_dataset: Ground truth dataset (optional)
        stats: Ensemble statistics (optional)
        metrics: Performance metrics (optional)
        config: Configuration dictionary (optional)
        output_dir: Directory to save the report
        verbose: Enable verbose output
        
    Returns:
        Path to saved report file
    """
    if verbose:
        print('📋 Creating summary report...')
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 4, figure=fig)
    
    # Title
    fig.suptitle('CorrDiff Ensemble Generation - Summary Report', 
                fontsize=18, fontweight='bold')
    
    # Configuration summary (text)
    ax_config = fig.add_subplot(gs[0, :2])
    ax_config.axis('off')
    
    config_text = "CONFIGURATION:\n"
    if config:
        config_text += f"Variables: {config.get('variables', 'N/A')}\n"
        config_text += f"Ensemble members: {config.get('num_ensembles', 'N/A')}\n"
        config_text += f"Sampling mode: {config.get('sampling_mode', 'N/A')}\n"
        config_text += f"Diffusion steps: {config.get('number_of_steps', 'N/A')}\n"
    
    # Dataset info
    dataset_text = "\nDATASET INFO:\n"
    dataset_text += f"Prediction shape: {dict(pred_dataset.dims)}\n"
    if truth_dataset:
        dataset_text += f"Truth shape: {dict(truth_dataset.dims)}\n"
    
    full_text = config_text + dataset_text
    ax_config.text(0.05, 0.95, full_text, transform=ax_config.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Sample visualization for first variable
    if pred_dataset.data_vars:
        var_name = list(pred_dataset.data_vars.keys())[0]
        var_data = pred_dataset[var_name]
        
        if 'ensemble' in var_data.dims:
            # Ensemble mean
            ax_mean = fig.add_subplot(gs[0, 2])
            ens_mean = var_data.mean(dim='ensemble')
            if 'time' in ens_mean.dims:
                ens_mean = ens_mean.isel(time=0)
            im1 = ax_mean.imshow(ens_mean, cmap='viridis', origin='lower')
            ax_mean.set_title(f'{var_name} - Ensemble Mean')
            ax_mean.set_xticks([])
            ax_mean.set_yticks([])
            
            # Ensemble std
            ax_std = fig.add_subplot(gs[0, 3])
            ens_std = var_data.std(dim='ensemble')
            if 'time' in ens_std.dims:
                ens_std = ens_std.isel(time=0)
            im2 = ax_std.imshow(ens_std, cmap='Reds', origin='lower')
            ax_std.set_title(f'{var_name} - Ensemble Std')
            ax_std.set_xticks([])
            ax_std.set_yticks([])
    
    # Metrics summary
    if metrics:
        ax_metrics = fig.add_subplot(gs[1, :2])
        
        metric_names = []
        metric_values = []
        
        for var_name, var_metrics in metrics.items():
            for metric_name, metric_data in var_metrics.items():
                if metric_name in ['rmse', 'correlation', 'r2', 'bias']:
                    if hasattr(metric_data, 'mean'):
                        metric_names.append(f'{var_name}_{metric_name}')
                        metric_values.append(float(metric_data.mean().values))
        
        if metric_names:
            bars = ax_metrics.bar(range(len(metric_names)), metric_values)
            ax_metrics.set_xticks(range(len(metric_names)))
            ax_metrics.set_xticklabels(metric_names, rotation=45, ha='right')
            ax_metrics.set_title('Key Performance Metrics')
            ax_metrics.grid(True, alpha=0.3)
    
    # Statistics summary
    if stats:
        ax_stats = fig.add_subplot(gs[1, 2:])
        
        for i, (var_name, var_stats) in enumerate(stats.items()):
            if i >= 3:  # Limit to 3 variables
                break
                
            if 'mean' in var_stats and 'std' in var_stats:
                mean_data = var_stats['mean']
                std_data = var_stats['std']
                
                # Spatial averages
                mean_spatial = mean_data.mean().values
                std_spatial = std_data.mean().values
                
                ax_stats.bar([f'{var_name}\nMean', f'{var_name}\nStd'], 
                           [mean_spatial, std_spatial], alpha=0.7)
        
        ax_stats.set_title('Ensemble Statistics Summary')
        ax_stats.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        filename = 'summary_report.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        if verbose:
            print(f'✅ Summary report saved: {filepath}')
        
        plt.show()
        plt.close()
        return filepath
    else:
        plt.show()
        plt.close()
        return None