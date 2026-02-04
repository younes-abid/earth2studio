"""
Statistics and analysis functions for CorrDiff ensemble results.
"""
import time
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def calculate_ensemble_statistics(dataset: xr.Dataset, variables: List[str], verbose: bool = True) -> Dict[str, xr.Dataset]:
    """
    Calculate comprehensive ensemble statistics.
    
    Args:
        dataset: xarray Dataset with ensemble dimension
        variables: List of variables to analyze
        verbose: Enable verbose output
        
    Returns:
        Dictionary of statistics datasets
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print('📊 Computing ensemble statistics...')
    
    stats = {}
    
    for var_name in variables:
        if var_name not in dataset:
            if verbose:
                print(f'   ⚠️  Variable {var_name} not found in dataset')
            continue
            
        var_data = dataset[var_name]  # Shape: (ensemble, time, y, x)
        
        if verbose:
            print(f'   Processing {var_name} with shape {var_data.shape}')
        
        # Compute ensemble statistics
        var_stats = xr.Dataset({
            'mean': var_data.mean(dim='ensemble'),
            'std': var_data.std(dim='ensemble'),
            'min': var_data.min(dim='ensemble'),
            'max': var_data.max(dim='ensemble'),
            'median': var_data.median(dim='ensemble'),
            'q25': var_data.quantile(0.25, dim='ensemble'),
            'q75': var_data.quantile(0.75, dim='ensemble'),
            'q10': var_data.quantile(0.10, dim='ensemble'),
            'q90': var_data.quantile(0.90, dim='ensemble'),
            'range': var_data.max(dim='ensemble') - var_data.min(dim='ensemble'),
            'iqr': var_data.quantile(0.75, dim='ensemble') - var_data.quantile(0.25, dim='ensemble'),
            'skewness': xr.apply_ufunc(
                lambda x: xr.apply_ufunc(np.mean, ((x - x.mean(dim='ensemble')) / x.std(dim='ensemble'))**3, dim='ensemble'),
                var_data,
                dask='allowed'
            ),
            'kurtosis': xr.apply_ufunc(
                lambda x: xr.apply_ufunc(np.mean, ((x - x.mean(dim='ensemble')) / x.std(dim='ensemble'))**4, dim='ensemble'),
                var_data,
                dask='allowed'
            )
        })
        
        # Copy coordinates
        var_stats = var_stats.assign_coords(var_data.drop_vars('ensemble').coords)
        
        stats[var_name] = var_stats
        
        if verbose:
            mean_val = var_stats['mean'].mean().values
            std_val = var_stats['std'].mean().values
            print(f'     Mean: {mean_val:.4f}, Avg uncertainty: {std_val:.4f}')
    
    if verbose:
        elapsed = time.time() - start_time
        print(f'✅ Ensemble statistics computed in {elapsed:.3f}s')
    
    return stats


def calculate_metrics_vs_truth(pred_dataset: xr.Dataset, truth_dataset: xr.Dataset, 
                              variables: List[str], verbose: bool = True) -> Dict[str, Dict[str, xr.DataArray]]:
    """
    Calculate performance metrics against ground truth.
    
    Args:
        pred_dataset: Prediction dataset with ensemble dimension
        truth_dataset: Ground truth dataset
        variables: List of variables to analyze
        verbose: Enable verbose output
        
    Returns:
        Dictionary of metrics by variable
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print('📏 Computing metrics vs ground truth...')
    
    metrics = {}
    
    for var_name in variables:
        if var_name not in pred_dataset or var_name not in truth_dataset:
            if verbose:
                print(f'   ⚠️  Variable {var_name} missing in prediction or truth dataset')
            continue
        
        pred_var = pred_dataset[var_name]  # (ensemble, time, y, x)
        truth_var = truth_dataset[var_name]  # (time, y, x)
        
        # Use ensemble mean for metrics
        pred_mean = pred_var.mean(dim='ensemble')
        
        if verbose:
            print(f'   Computing metrics for {var_name}')
        
        # Calculate metrics
        diff = pred_mean - truth_var
        squared_diff = diff ** 2
        abs_diff = np.abs(diff)
        
        # RMSE
        rmse = np.sqrt(squared_diff.mean(dim=['y', 'x']))
        
        # MAE
        mae = abs_diff.mean(dim=['y', 'x'])
        
        # Bias
        bias = diff.mean(dim=['y', 'x'])
        
        # Correlation
        def correlation_2d(pred, truth):
            """Calculate correlation for 2D fields."""
            pred_flat = pred.stack(spatial=['y', 'x'])
            truth_flat = truth.stack(spatial=['y', 'x'])
            return xr.corr(pred_flat, truth_flat, dim='spatial')
        
        correlation = xr.apply_ufunc(
            correlation_2d,
            pred_mean,
            truth_var,
            input_core_dims=[['y', 'x'], ['y', 'x']],
            vectorize=True,
            dask='allowed'
        )
        
        # R²
        ss_res = squared_diff.sum(dim=['y', 'x'])
        ss_tot = ((truth_var - truth_var.mean(dim=['y', 'x']))**2).sum(dim=['y', 'x'])
        r2 = 1 - (ss_res / ss_tot)
        
        # Normalized metrics
        truth_std = truth_var.std(dim=['y', 'x'])
        nrmse = rmse / truth_std
        nmae = mae / truth_var.mean(dim=['y', 'x'])
        
        # Ensemble-specific metrics
        ensemble_rmse = np.sqrt(((pred_var - truth_var)**2).mean(dim=['y', 'x']))
        ensemble_spread = pred_var.std(dim='ensemble').mean(dim=['y', 'x'])
        
        # Reliability metrics
        ens_var = pred_var.var(dim='ensemble')
        mse = squared_diff.mean(dim=['y', 'x'])
        reliability = ens_var.mean(dim=['y', 'x']) - mse
        
        metrics[var_name] = {
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'correlation': correlation,
            'r2': r2,
            'nrmse': nrmse,
            'nmae': nmae,
            'ensemble_rmse': ensemble_rmse,
            'ensemble_spread': ensemble_spread,
            'reliability': reliability
        }
        
        if verbose:
            print(f'     RMSE: {rmse.mean().values:.4f}')
            print(f'     Correlation: {correlation.mean().values:.4f}')
            print(f'     R²: {r2.mean().values:.4f}')
    
    if verbose:
        elapsed = time.time() - start_time
        print(f'✅ Metrics computed in {elapsed:.3f}s')
    
    return metrics


def calculate_spatial_statistics(dataset: xr.Dataset, variables: List[str], 
                                verbose: bool = True) -> Dict[str, Dict[str, xr.DataArray]]:
    """
    Calculate spatial statistics (averages over space).
    
    Args:
        dataset: xarray Dataset
        variables: List of variables to analyze
        verbose: Enable verbose output
        
    Returns:
        Dictionary of spatial statistics
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print('🗺️  Computing spatial statistics...')
    
    spatial_stats = {}
    
    for var_name in variables:
        if var_name not in dataset:
            continue
            
        var_data = dataset[var_name]
        
        # Calculate spatial means and other statistics
        spatial_stats[var_name] = {
            'spatial_mean': var_data.mean(dim=['y', 'x']),
            'spatial_std': var_data.std(dim=['y', 'x']),
            'spatial_min': var_data.min(dim=['y', 'x']),
            'spatial_max': var_data.max(dim=['y', 'x']),
            'spatial_median': var_data.median(dim=['y', 'x'])
        }
        
        if verbose:
            mean_val = spatial_stats[var_name]['spatial_mean'].mean().values
            print(f'   {var_name} spatial average: {mean_val:.4f}')
    
    if verbose:
        elapsed = time.time() - start_time
        print(f'✅ Spatial statistics computed in {elapsed:.3f}s')
    
    return spatial_stats


def calculate_temporal_statistics(dataset: xr.Dataset, variables: List[str],
                                 verbose: bool = True) -> Dict[str, Dict[str, xr.DataArray]]:
    """
    Calculate temporal statistics (averages over time).
    
    Args:
        dataset: xarray Dataset
        variables: List of variables to analyze
        verbose: Enable verbose output
        
    Returns:
        Dictionary of temporal statistics
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print('⏰ Computing temporal statistics...')
    
    temporal_stats = {}
    
    for var_name in variables:
        if var_name not in dataset:
            continue
            
        var_data = dataset[var_name]
        
        if 'time' not in var_data.dims:
            if verbose:
                print(f'   ⚠️  No time dimension for {var_name}')
            continue
        
        # Calculate temporal means and other statistics
        temporal_stats[var_name] = {
            'temporal_mean': var_data.mean(dim='time'),
            'temporal_std': var_data.std(dim='time'),
            'temporal_min': var_data.min(dim='time'),
            'temporal_max': var_data.max(dim='time'),
            'temporal_trend': calculate_temporal_trend(var_data)
        }
        
        if verbose:
            print(f'   {var_name} temporal statistics computed')
    
    if verbose:
        elapsed = time.time() - start_time
        print(f'✅ Temporal statistics computed in {elapsed:.3f}s')
    
    return temporal_stats


def calculate_temporal_trend(data: xr.DataArray) -> xr.DataArray:
    """Calculate linear trend over time dimension."""
    if 'time' not in data.dims:
        return xr.zeros_like(data.isel(time=0) if 'time' in data.coords else data)
    
    # Convert time to numeric (assuming it's datetime)
    time_numeric = (data.time - data.time[0]) / np.timedelta64(1, 'h')  # hours
    
    # Calculate linear trend using least squares
    def linear_trend(y, x):
        """Calculate slope of linear trend."""
        if len(y) < 2:
            return 0.0
        n = len(y)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    # Apply trend calculation
    trend = xr.apply_ufunc(
        linear_trend,
        data,
        time_numeric,
        input_core_dims=[['time'], ['time']],
        vectorize=True,
        dask='allowed'
    )
    
    return trend


def analyze_ensemble_reliability(pred_dataset: xr.Dataset, truth_dataset: xr.Dataset,
                               variables: List[str], verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Analyze ensemble reliability and calibration.
    
    Args:
        pred_dataset: Prediction dataset with ensemble dimension
        truth_dataset: Ground truth dataset
        variables: List of variables to analyze
        verbose: Enable verbose output
        
    Returns:
        Dictionary of reliability analysis results
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print('🎯 Analyzing ensemble reliability...')
    
    reliability = {}
    
    for var_name in variables:
        if var_name not in pred_dataset or var_name not in truth_dataset:
            continue
            
        pred_var = pred_dataset[var_name]  # (ensemble, time, y, x)
        truth_var = truth_dataset[var_name]  # (time, y, x)
        
        if verbose:
            print(f'   Analyzing {var_name}')
        
        # Calculate ensemble statistics
        ens_mean = pred_var.mean(dim='ensemble')
        ens_var = pred_var.var(dim='ensemble')
        
        # Mean squared error
        mse = ((ens_mean - truth_var)**2).mean()
        
        # Ensemble variance (average over space and time)
        avg_ens_var = ens_var.mean()
        
        # Reliability: how well ensemble spread matches actual error
        reliability_metric = avg_ens_var.values - mse.values
        
        # Rank histogram analysis
        ranks = calculate_rank_histogram(pred_var, truth_var)
        
        # Coverage analysis for different prediction intervals
        coverage = calculate_prediction_interval_coverage(pred_var, truth_var)
        
        reliability[var_name] = {
            'reliability_metric': reliability_metric,
            'mse': mse.values,
            'ensemble_variance': avg_ens_var.values,
            'rank_histogram': ranks,
            'coverage': coverage
        }
        
        if verbose:
            print(f'     Reliability: {reliability_metric:.4f}')
            print(f'     MSE: {mse.values:.4f}')
            print(f'     Ensemble variance: {avg_ens_var.values:.4f}')
    
    if verbose:
        elapsed = time.time() - start_time
        print(f'✅ Reliability analysis completed in {elapsed:.3f}s')
    
    return reliability


def calculate_rank_histogram(pred_var: xr.DataArray, truth_var: xr.DataArray, 
                           n_bins: int = None) -> np.ndarray:
    """Calculate rank histogram for ensemble calibration."""
    n_ensemble = pred_var.sizes['ensemble']
    if n_bins is None:
        n_bins = n_ensemble + 1
    
    # Flatten spatial and temporal dimensions
    pred_flat = pred_var.stack(flat=['time', 'y', 'x']).transpose('flat', 'ensemble')
    truth_flat = truth_var.stack(flat=['time', 'y', 'x'])
    
    ranks = np.zeros(n_bins)
    
    for i in range(len(truth_flat)):
        ensemble_values = pred_flat[i].values
        truth_value = truth_flat[i].values
        
        # Calculate rank of truth value among ensemble members
        rank = np.sum(ensemble_values < truth_value)
        rank = min(rank, n_bins - 1)  # Cap at max rank
        ranks[rank] += 1
    
    # Normalize
    ranks = ranks / np.sum(ranks)
    
    return ranks


def calculate_prediction_interval_coverage(pred_var: xr.DataArray, truth_var: xr.DataArray) -> Dict[str, float]:
    """Calculate prediction interval coverage for different confidence levels."""
    coverage = {}
    
    confidence_levels = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
    
    for conf in confidence_levels:
        alpha = (1 - conf) / 2
        lower_quantile = alpha
        upper_quantile = 1 - alpha
        
        lower_bound = pred_var.quantile(lower_quantile, dim='ensemble')
        upper_bound = pred_var.quantile(upper_quantile, dim='ensemble')
        
        # Check if truth falls within prediction interval
        within_interval = (truth_var >= lower_bound) & (truth_var <= upper_bound)
        actual_coverage = within_interval.mean().values
        
        coverage[f'{conf*100:.0f}%'] = actual_coverage
    
    return coverage


def save_statistics_to_netcdf(stats: Dict, output_path: str, verbose: bool = True):
    """Save statistics to NetCDF file."""
    if verbose:
        print(f'💾 Saving statistics to {output_path}...')
    
    # Create combined dataset
    combined_stats = xr.Dataset()
    
    for var_name, var_stats in stats.items():
        if isinstance(var_stats, xr.Dataset):
            # Add variable name prefix to avoid conflicts
            for stat_name, stat_data in var_stats.data_vars.items():
                combined_stats[f'{var_name}_{stat_name}'] = stat_data
        elif isinstance(var_stats, dict):
            for stat_name, stat_data in var_stats.items():
                if isinstance(stat_data, xr.DataArray):
                    combined_stats[f'{var_name}_{stat_name}'] = stat_data
    
    # Save to file
    combined_stats.to_netcdf(output_path)
    
    if verbose:
        print(f'✅ Statistics saved to {output_path}')


def print_summary_statistics(stats: Dict, metrics: Dict = None, verbose: bool = True):
    """Print summary of computed statistics."""
    if not verbose:
        return
        
    print('\n' + '='*60)
    print('📊 STATISTICAL ANALYSIS SUMMARY')
    print('='*60)
    
    for var_name in stats.keys():
        print(f'\n{var_name}:')
        
        if var_name in stats:
            var_stats = stats[var_name]
            if isinstance(var_stats, dict):
                for stat_name, stat_data in var_stats.items():
                    if isinstance(stat_data, (xr.DataArray, xr.Dataset)):
                        mean_val = float(stat_data.mean().values) if hasattr(stat_data, 'mean') else 'N/A'
                        print(f'  {stat_name}: {mean_val:.4f}')
                    else:
                        print(f'  {stat_name}: {stat_data}')
        
        if metrics and var_name in metrics:
            print('  Performance Metrics:')
            var_metrics = metrics[var_name]
            for metric_name, metric_data in var_metrics.items():
                if isinstance(metric_data, xr.DataArray):
                    mean_val = float(metric_data.mean().values)
                    print(f'    {metric_name}: {mean_val:.4f}')
    
    print('='*60)