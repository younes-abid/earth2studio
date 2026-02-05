# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Visualization orchestrator that manages all plot generation based on configuration.
Minimal implementation focused on prediction plots with placeholders for future extensions.
"""

import os
from pathlib import Path
import xarray as xr
import pandas as pd
from datetime import datetime


def _get_timestamp_from_netcdf(netcdf_path, time_idx):
    """
    Extract timestamp from NetCDF file at given time index.
    
    Parameters
    ----------
    netcdf_path : str
        Path to NetCDF file
    time_idx : int
        Time index to extract
        
    Returns
    -------
    tuple
        (timestamp_str, sanitized_timestamp_str)
    """
    try:
        ds = xr.open_dataset(netcdf_path)
        
        # Find time coordinate (look for 'time' or datetime-like coordinates)
        time_coord_name = None
        for coord_name in ds.coords:
            if 'time' in coord_name.lower() or ds.coords[coord_name].dtype.kind == 'M':
                time_coord_name = coord_name
                break
        
        if time_coord_name is None:
            ds.close()
            return "unknown_time", "unknown_time"
        
        # Get timestamp at time_idx
        time_coord = ds.coords[time_coord_name]
        if time_idx < len(time_coord):
            timestamp = pd.to_datetime(time_coord.values[time_idx])
            timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S')
            sanitized_str = timestamp.strftime('%Y-%m-%dT%H-%M-%S')
            ds.close()
            return timestamp_str, sanitized_str
        else:
            ds.close()
            return "invalid_time_idx", "invalid_time_idx"
            
    except Exception as e:
        print(f"   ⚠️ Warning: Could not extract timestamp from NetCDF: {e}")
        return "unknown_time", "unknown_time"


def _create_timestamped_output_folder(base_output_folder, time_idx, netcdf_path):
    """
    Create output folder with timestamp structure.
    
    Parameters
    ----------
    base_output_folder : Path
        Base output folder path
    time_idx : int
        Time index
    netcdf_path : str
        Path to NetCDF file
        
    Returns
    -------
    Path
        Path to timestamped folder
    """
    # Get timestamp from NetCDF
    timestamp_str, sanitized_timestamp = _get_timestamp_from_netcdf(netcdf_path, time_idx)
    
    # Create timestamped folder name: 001_2022-07-27T00-00-00
    timestamp_folder = f"{time_idx+1:03d}_{sanitized_timestamp}"
    
    # Create full path
    timestamped_folder = base_output_folder / timestamp_folder
    timestamped_folder.mkdir(parents=True, exist_ok=True)
    
    return timestamped_folder, sanitized_timestamp


def plot_analysis(config, netcdf_path, time_idx=0, show=True, variable_idx=None):
    """
    Main function to generate all visualization plots based on configuration.
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration with plot flags and settings
    netcdf_path : str
        Path to the NetCDF ensemble file
    time_idx : int
        Time index to plot
    show : bool
        Whether to print summary
    variable_idx : int or None
        Variable index to plot (if None, plots all variables)
        
    Returns
    -------
    dict
        Dictionary mapping plot types to saved file paths
    """
    if show:
        print("🎨 Starting visualization generation...")
    
    # Ensure base output folder exists
    base_output_folder = Path(config.OUTPUT_ANALYSIS_FOLDER)
    base_output_folder.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped output folder
    timestamped_folder, sanitized_timestamp = _create_timestamped_output_folder(
        base_output_folder, time_idx, netcdf_path
    )
    
    if show:
        print(f"   📁 Output folder: {timestamped_folder}")
    
    results = {}
    
    # Determine which variables to plot
    if variable_idx is not None:
        # Plot single variable (backward compatibility)
        variable_indices = [variable_idx]
        if show:
            var_name = config.OUTPUT_VARIABLES[variable_idx] if hasattr(config, 'OUTPUT_VARIABLES') else f"Variable {variable_idx}"
            print(f"   📊 Processing single variable: {var_name}")
    else:
        # Plot all variables (new default behavior)
        num_variables = len(getattr(config, 'OUTPUT_VARIABLES', []))
        if num_variables == 0:
            print("   ⚠️ No OUTPUT_VARIABLES found in config, defaulting to variable_idx=0")
            variable_indices = [0]
        else:
            variable_indices = list(range(num_variables))
            if show:
                print(f"   📊 Processing {num_variables} variables: {config.OUTPUT_VARIABLES}")
    
    # Generate plots for each variable
    for var_idx in variable_indices:
        var_name = config.OUTPUT_VARIABLES[var_idx] if hasattr(config, 'OUTPUT_VARIABLES') and var_idx < len(config.OUTPUT_VARIABLES) else f"var_{var_idx}"
        
        if show and len(variable_indices) > 1:
            print(f"\n   🔄 Processing variable {var_idx + 1}/{len(variable_indices)}: {var_name}")
        
        # Generate prediction plots for this variable
        prediction_results = _generate_prediction_plots(
            config, netcdf_path, time_idx, var_idx, timestamped_folder, var_name, sanitized_timestamp
        )
        results.update(prediction_results)
        
        # Generate core metrics for this variable
        core_metrics_results = _generate_core_metrics(
            config, netcdf_path, time_idx, var_idx, timestamped_folder, var_name, sanitized_timestamp
        )
        results.update(core_metrics_results)
    
    # Placeholders for future implementation
    if hasattr(config, 'PLOT_INPUT_VARIABLES') and config.PLOT_INPUT_VARIABLES:
        if show:
            print("   🔄 Input plots - placeholder (not implemented yet)")
    
    if any(getattr(config, attr, None) for attr in ['PLOT_CORE_METRICS', 'PLOT_IMAGE_QUALITY_METRICS', 
                                                   'PLOT_ENSEMBLE_METRICS', 'PLOT_DIFFUSION_METRICS', 
                                                   'PLOT_SPECTRAL_METRICS', 'PLOT_PATTERN_METRICS']):
        if show:
            print("   🔄 Metric plots - placeholder (not implemented yet)")
    
    if show:
        print(f"✅ Generated {len(results)} plot files")
        _print_summary(results)
    
    return results


def _generate_prediction_plots(config, netcdf_path, time_idx, variable_idx, output_folder, var_name=None, sanitized_timestamp=None):
    """Generate prediction plots based on configuration flags"""
    results = {}
    prediction_folder = output_folder / "prediction"
    prediction_folder.mkdir(parents=True, exist_ok=True)
    
    # Get variable name for file naming
    if var_name is None:
        var_name = config.OUTPUT_VARIABLES[variable_idx] if hasattr(config, 'OUTPUT_VARIABLES') and variable_idx < len(config.OUTPUT_VARIABLES) else f"var_{variable_idx}"
    
    # Prediction plot configurations: (config_flag, module_name, class_name)
    plot_configs = [
        ('PLOT_ENSEMBLE_MEMBERS', 'ensemble_members', 'EnsembleMembersPlot'),
        ('PLOT_ENSEMBLE_STATISTICS', 'ensemble_statistics', 'EnsembleStatisticsPlot'), 
        ('PLOT_ENSEMBLE_RESIDUALS', 'ensemble_residuals', 'EnsembleResidualsPlot'),
        ('PLOT_ENSEMBLE_RESIDUALS_STATISTICS', 'residuals_statistics', 'ResidualsStatisticsPlot'),
        ('PLOT_GROUND_TRUTH_VS_ENSEMBLEMEAN', 'truth_vs_mean', 'TruthVsMeanPlot'),
        ('PLOT_UNCERTAINTY_QUANTIFICATION', 'uncertainty_quantification', 'UncertaintyQuantificationPlot')
    ]
    
    for config_flag, module_name, class_name in plot_configs:
        if getattr(config, config_flag, False):
            try:
                # Dynamic import of prediction modules
                module = __import__(f'visualization.prediction.{module_name}', fromlist=[class_name])
                plot_class = getattr(module, class_name)
                
                # Create plotter and generate plot
                plotter = plot_class(config, prediction_folder)
                result = plotter.plot(netcdf_path, time_idx=time_idx, variable_idx=variable_idx)
                
                if result:
                    # Include variable name in key for multiple variables
                    key = f"{module_name}_{var_name}" if len(getattr(config, 'OUTPUT_VARIABLES', [])) > 1 else module_name
                    results[key] = result
                    
            except Exception as e:
                print(f"   ⚠️ Error generating {module_name} for {var_name}: {e}")
    
    return results


def _generate_core_metrics(config, netcdf_path, time_idx, variable_idx, output_folder, var_name=None, sanitized_timestamp=None):
    """Generate core regression metrics based on configuration flags"""
    results = {}
    
    if not config.PLOT_CORE_METRICS:
        return results
    
    # Get variable name for file naming
    if var_name is None:
        var_name = config.OUTPUT_VARIABLES[variable_idx] if hasattr(config, 'OUTPUT_VARIABLES') and variable_idx < len(config.OUTPUT_VARIABLES) else f"var_{variable_idx}"
    
    core_folder = output_folder / "core_metrics"
    core_folder.mkdir(parents=True, exist_ok=True)
    
    # Core metric configurations: (metric_name, module_name, class_name)
    metric_configs = [
        ('RMSE', 'rmse', 'RMSEPlot'),
        ('MAE', 'mae', 'MAEPlot'),
        ('R2', 'r2', 'R2Plot'),
        ('BIAS', 'bias', 'BIASPlot'),
        ('MAPE', 'mape', 'MAPEPlot')
    ]
    
    for metric_name, module_name, class_name in metric_configs:
        if metric_name in config.PLOT_CORE_METRICS:
            try:
                # Dynamic import of core metric modules
                module = __import__(f'visualization.metrics.core.{module_name}', fromlist=[class_name])
                metric_class = getattr(module, class_name)
                
                # Get metric-specific config
                metric_config = config.PLOT_CORE_METRICS.get(metric_name, {})
                
                # Create metric plotter and generate plot
                plotter = metric_class(config, core_folder, metric_name, metric_config)
                result = plotter.plot(netcdf_path, time_idx=time_idx, variable_idx=variable_idx, **metric_config)
                
                if result:
                    # Include variable name in key for multiple variables
                    key = f"{metric_name.lower()}_metric_{var_name}" if len(getattr(config, 'OUTPUT_VARIABLES', [])) > 1 else f"{metric_name.lower()}_metric"
                    results[key] = result
                    print(f"   ✅ {metric_name} for {var_name}")
                    
            except Exception as e:
                print(f"   ⚠️ Error generating {metric_name} for {var_name}: {e}")
    
    return results


def _print_summary(results):
    """Print summary of generated plots"""
    print("\n📊 Visualization Summary:")
    print("-" * 50)
    for plot_type, file_path in results.items():
        filename = os.path.basename(file_path) if file_path else "Failed"
        print(f"✅ {plot_type:30} → {filename}")
    print("-" * 50)


class VisualizationOrchestrator:
    """
    Legacy class for backward compatibility.
    Use plot_analysis() function instead.
    """
    
    def __init__(self, config, output_folder):
        self.config = config
        self.output_folder = output_folder
    
    def generate_all_plots(self, netcdf_path, time_idx=0, variable_idx=0):
        """Legacy method - use plot_analysis() instead"""
        return plot_analysis(self.config, netcdf_path, time_idx, show=False, variable_idx=variable_idx)