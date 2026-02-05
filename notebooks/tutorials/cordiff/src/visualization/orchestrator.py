# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Visualization orchestrator that manages all plot generation based on configuration.
Minimal implementation focused on prediction plots with placeholders for future extensions.
"""

import os
from pathlib import Path


def plot_analysis(config, netcdf_path, time_idx=0, show=True, variable_idx=0):
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
    variable_idx : int
        Variable index to plot (if multiple variables)
        
    Returns
    -------
    dict
        Dictionary mapping plot types to saved file paths
    """
    if show:
        print("🎨 Starting visualization generation...")
    
    # Ensure output folder exists
    output_folder = Path(config.OUTPUT_ANALYSIS_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Generate prediction plots
    prediction_results = _generate_prediction_plots(config, netcdf_path, time_idx, variable_idx, output_folder)
    results.update(prediction_results)
    
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


def _generate_prediction_plots(config, netcdf_path, time_idx, variable_idx, output_folder):
    """Generate prediction plots based on configuration flags"""
    results = {}
    prediction_folder = output_folder / "prediction"
    prediction_folder.mkdir(parents=True, exist_ok=True)
    
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
                    results[module_name] = result
                    print(f"   ✅ {module_name}")
                    
            except Exception as e:
                print(f"   ⚠️ Error generating {module_name}: {e}")
    
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