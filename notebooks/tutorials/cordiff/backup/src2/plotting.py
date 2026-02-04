"""
Simple plotting utilities for ensemble analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import Dict, List, Optional
from pathlib import Path


def plot_ensemble_analysis(datasets: Dict[str, xr.Dataset], output_dir: str, 
                          variables: List[str], verbose: bool = True):
    """Create comprehensive ensemble analysis plots."""
    if verbose:
        print('📊 Creating ensemble analysis plots...')
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pred_ds = datasets['prediction']
    has_truth = 'truth' in datasets and len(datasets['truth'].data_vars) > 0
    
    for var_name in variables:
        if var_name not in pred_ds.data_vars:
            if verbose:
                print(f'   ⚠️  Variable {var_name} not found in predictions')
            continue
            
        if verbose:
            print(f'   🎨 Plotting {var_name}...')
        
        var_data = pred_ds[var_name]
        
        # 1. All ensemble members plot
        plot_all_ensemble_members(var_data, var_name, output_dir, has_truth, datasets)
        
        # 2. Ensemble mean and spread
        plot_ensemble_statistics(var_data, var_name, output_dir, has_truth, datasets)
        
        # 3. Histogram of ensemble values
        plot_ensemble_histograms(var_data, var_name, output_dir)
        
        # 4. Metrics plot (if ground truth available)
        if has_truth:
            plot_ensemble_metrics(var_data, datasets['truth'][var_name], var_name, output_dir)
    
    if verbose:
        print(f'✅ Plots saved to: {output_dir}')


def plot_all_ensemble_members(var_data: xr.DataArray, var_name: str, output_dir: str,
                             has_truth: bool, datasets: Dict):
    """Plot all ensemble members for each time step."""
    n_times = len(var_data.time)
    n_ensembles = len(var_data.ensemble)
    
    for t_idx, time_val in enumerate(var_data.time):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'{var_name} - All Ensemble Members - {str(time_val.values)[:19]}', fontsize=14)
        
        axes = axes.flatten()
        
        # Plot first 8 ensemble members
        for ens_idx in range(min(8, n_ensembles)):
            ax = axes[ens_idx]
            data_slice = var_data.isel(time=t_idx, ensemble=ens_idx)
            
            im = ax.imshow(data_slice.values, cmap='viridis', aspect='auto')
            ax.set_title(f'Member {ens_idx+1}')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Hide unused subplots
        for i in range(n_ensembles, 8):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{var_name}_all_ensemble_t{t_idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_ensemble_statistics(var_data: xr.DataArray, var_name: str, output_dir: str,
                           has_truth: bool, datasets: Dict):
    """Plot ensemble mean, spread, and optionally ground truth."""
    n_times = len(var_data.time)
    
    # Calculate ensemble statistics
    ens_mean = var_data.mean(dim='ensemble')
    ens_std = var_data.std(dim='ensemble')
    
    for t_idx, time_val in enumerate(var_data.time):
        n_cols = 3 if has_truth else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4))
        if n_cols == 1:
            axes = [axes]
        
        fig.suptitle(f'{var_name} - Ensemble Statistics - {str(time_val.values)[:19]}', fontsize=14)
        
        # Ensemble mean
        mean_slice = ens_mean.isel(time=t_idx)
        im1 = axes[0].imshow(mean_slice.values, cmap='viridis', aspect='auto')
        axes[0].set_title('Ensemble Mean')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Ensemble standard deviation
        std_slice = ens_std.isel(time=t_idx)
        im2 = axes[1].imshow(std_slice.values, cmap='Reds', aspect='auto')
        axes[1].set_title('Ensemble Std Dev (Uncertainty)')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Ground truth (if available)
        if has_truth and var_name in datasets['truth'].data_vars:
            truth_slice = datasets['truth'][var_name].isel(time=t_idx)
            im3 = axes[2].imshow(truth_slice.values, cmap='viridis', aspect='auto')
            axes[2].set_title('Ground Truth')
            axes[2].set_xlabel('Longitude')
            axes[2].set_ylabel('Latitude')
            plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{var_name}_statistics_t{t_idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_ensemble_histograms(var_data: xr.DataArray, var_name: str, output_dir: str):
    """Plot histograms of ensemble values for each time step."""
    n_times = len(var_data.time)
    
    for t_idx, time_val in enumerate(var_data.time):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{var_name} - Ensemble Value Distributions - {str(time_val.values)[:19]}', fontsize=14)
        
        time_slice = var_data.isel(time=t_idx)
        
        # Overall histogram
        axes[0, 0].hist(time_slice.values.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('All Ensemble Values')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Per-ensemble histograms
        axes[0, 1].set_title('Per-Ensemble Distributions')
        colors = plt.cm.tab10(np.linspace(0, 1, len(time_slice.ensemble)))
        for ens_idx, color in zip(range(len(time_slice.ensemble)), colors):
            ens_values = time_slice.isel(ensemble=ens_idx).values.flatten()
            axes[0, 1].hist(ens_values, bins=30, alpha=0.5, color=color, 
                          label=f'Ens {ens_idx+1}', density=True)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Ensemble mean vs std scatter
        ens_means = time_slice.mean(dim=['y', 'x'])
        ens_stds = time_slice.std(dim=['y', 'x'])
        axes[1, 0].scatter(ens_means.values, ens_stds.values, alpha=0.7, s=50)
        axes[1, 0].set_title('Ensemble Mean vs Std (Spatial)')
        axes[1, 0].set_xlabel('Spatial Mean')
        axes[1, 0].set_ylabel('Spatial Std')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot of ensemble ranges
        box_data = [time_slice.isel(ensemble=i).values.flatten() for i in range(len(time_slice.ensemble))]
        axes[1, 1].boxplot(box_data, labels=[f'E{i+1}' for i in range(len(time_slice.ensemble))])
        axes[1, 1].set_title('Ensemble Value Ranges')
        axes[1, 1].set_xlabel('Ensemble Member')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{var_name}_histograms_t{t_idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_ensemble_metrics(pred_data: xr.DataArray, truth_data: xr.DataArray, 
                         var_name: str, output_dir: str):
    """Plot metrics comparing ensemble predictions to ground truth."""
    n_times = len(pred_data.time)
    
    # Calculate metrics for each time step
    bias_values = []
    rmse_values = []
    mae_values = []
    spread_values = []
    
    for t_idx in range(n_times):
        pred_slice = pred_data.isel(time=t_idx)
        truth_slice = truth_data.isel(time=t_idx)
        
        # Ensemble mean
        ens_mean = pred_slice.mean(dim='ensemble')
        
        # Metrics
        bias = (ens_mean - truth_slice).mean().values
        rmse = np.sqrt(((ens_mean - truth_slice)**2).mean().values)
        mae = np.abs(ens_mean - truth_slice).mean().values
        spread = pred_slice.std(dim='ensemble').mean().values
        
        bias_values.append(bias)
        rmse_values.append(rmse)
        mae_values.append(mae)
        spread_values.append(spread)
    
    # Create metrics plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{var_name} - Ensemble Performance Metrics', fontsize=14)
    
    time_indices = range(n_times)
    
    # Bias
    axes[0, 0].plot(time_indices, bias_values, 'o-', color='red', linewidth=2)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Bias (Ensemble Mean - Truth)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Bias')
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE
    axes[0, 1].plot(time_indices, rmse_values, 'o-', color='blue', linewidth=2)
    axes[0, 1].set_title('Root Mean Square Error')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAE
    axes[1, 0].plot(time_indices, mae_values, 'o-', color='green', linewidth=2)
    axes[1, 0].set_title('Mean Absolute Error')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Spread vs RMSE
    axes[1, 1].scatter(spread_values, rmse_values, alpha=0.7, s=50, color='purple')
    axes[1, 1].plot([0, max(max(spread_values), max(rmse_values))], 
                   [0, max(max(spread_values), max(rmse_values))], 
                   'k--', alpha=0.5, label='Perfect Spread')
    axes[1, 1].set_title('Ensemble Spread vs RMSE')
    axes[1, 1].set_xlabel('Ensemble Spread')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{var_name}_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary metrics
    print(f'   📊 {var_name} Metrics Summary:')
    print(f'      Mean Bias: {np.mean(bias_values):.4f} ± {np.std(bias_values):.4f}')
    print(f'      Mean RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}')
    print(f'      Mean MAE:  {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}')
    print(f'      Mean Spread: {np.mean(spread_values):.4f} ± {np.std(spread_values):.4f}')


def create_summary_report(datasets: Dict[str, xr.Dataset], variables: List[str], 
                         output_dir: str, verbose: bool = True):
    """Create a summary report of the ensemble analysis."""
    if verbose:
        print('📋 Creating summary report...')
    
    pred_ds = datasets['prediction']
    has_truth = 'truth' in datasets and len(datasets['truth'].data_vars) > 0
    
    report_path = f'{output_dir}/ensemble_summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("ENSEMBLE ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Generated: {pred_ds.attrs.get('created', 'Unknown')}\n")
        f.write(f"Sampling Mode: {pred_ds.attrs.get('sampling_mode', 'Unknown')}\n")
        f.write(f"Number of Steps: {pred_ds.attrs.get('num_steps', 'Unknown')}\n")
        f.write(f"Ensemble Members: {len(pred_ds.ensemble)}\n")
        f.write(f"Time Steps: {len(pred_ds.time)}\n")
        f.write(f"Variables: {variables}\n")
        f.write(f"Ground Truth Available: {has_truth}\n\n")
        
        for var_name in variables:
            if var_name in pred_ds.data_vars:
                var_data = pred_ds[var_name]
                
                f.write(f"{var_name.upper()} STATISTICS:\n")
                f.write("-" * 30 + "\n")
                
                # Overall statistics
                ens_mean = var_data.mean(dim=['ensemble', 'time'])
                ens_std = var_data.std(dim=['ensemble', 'time'])
                ens_spread = var_data.std(dim='ensemble').mean()
                
                f.write(f"  Overall Mean: {ens_mean.values:.6f}\n")
                f.write(f"  Overall Std:  {ens_std.values:.6f}\n")
                f.write(f"  Avg Ensemble Spread: {ens_spread.values:.6f}\n")
                
                # Min/Max ranges
                var_min = var_data.min().values
                var_max = var_data.max().values
                f.write(f"  Value Range: [{var_min:.6f}, {var_max:.6f}]\n")
                
                f.write("\n")
        
        f.write("FILES GENERATED:\n")
        f.write("-" * 15 + "\n")
        for var_name in variables:
            f.write(f"  {var_name}:\n")
            f.write(f"    - *_all_ensemble_*.png (All ensemble members)\n")
            f.write(f"    - *_statistics_*.png (Mean, std, truth)\n")
            f.write(f"    - *_histograms_*.png (Value distributions)\n")
            if has_truth:
                f.write(f"    - *_metrics.png (Performance metrics)\n")
            f.write("\n")
    
    if verbose:
        print(f'📋 Summary report saved to: {report_path}')