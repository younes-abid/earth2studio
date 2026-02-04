"""
Main workflow orchestration for CorrDiff ensemble experiments.
"""
import os
import time
import torch
from typing import Dict, List, Optional, Any
from pathlib import Path

from .config import CorrDiffConfig
from .model_builder import load_models, initialize_model_with_data
from .data_utils import load_data_source, run_ensemble_inference, save_ensemble_netcdf, load_ensemble_results, validate_datasets
from .statistics import (
    calculate_ensemble_statistics, 
    calculate_metrics_vs_truth,
    calculate_spatial_statistics,
    analyze_ensemble_reliability,
    print_summary_statistics
)
from .visualization import create_comprehensive_visualization, create_summary_report


def run_complete_corrdiff_workflow(config: CorrDiffConfig) -> Dict[str, Any]:
    """
    Run the complete CorrDiff ensemble workflow.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with all results and paths
    """
    start_time = time.time()
    verbose = config.get('verbose', True)
    
    if verbose:
        print('🚀 Starting Complete CorrDiff Ensemble Workflow')
        print('=' * 60)
    
    # Validate configuration
    config.validate_config()
    if verbose:
        config.print_summary()
    
    results = {
        'config': config,
        'datasets': None,
        'ensemble_model': None,
        'inference_results': None,
        'statistics': None,
        'metrics': None,
        'visualization_files': None,
        'output_file': None,
        'plot_dir': None
    }
    
    try:
        # =================================================================
        # STEP 1: Load Data Source
        # =================================================================
        if verbose:
            print('\n📂 STEP 1: Loading Data Source')
            print('-' * 30)
        
        data_source = load_data_source(config, verbose)
        
        # =================================================================
        # STEP 2: Load and Initialize Models
        # =================================================================
        if verbose:
            print('\n🤖 STEP 2: Loading Models')
            print('-' * 30)
        
        ensemble_model = load_models(config, verbose)
        ensemble_model = initialize_model_with_data(ensemble_model, data_source, verbose)
        results['ensemble_model'] = ensemble_model
        
        # =================================================================
        # STEP 3: Run Ensemble Inference
        # =================================================================
        if verbose:
            print('\n🔄 STEP 3: Running Ensemble Inference')
            print('-' * 30)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_results = run_ensemble_inference(
            config.get('inference_times'), 
            ensemble_model, 
            data_source, 
            device, 
            verbose
        )
        results['inference_results'] = inference_results
        
        # =================================================================
        # STEP 4: Save Results
        # =================================================================
        if verbose:
            print('\n💾 STEP 4: Saving Results')
            print('-' * 30)
        
        output_file, plot_dir = config.get_output_paths()
        saved_file = save_ensemble_netcdf(inference_results, config, output_file, verbose)
        results['output_file'] = saved_file
        results['plot_dir'] = plot_dir
        
        # =================================================================
        # STEP 5: Load and Validate Results
        # =================================================================
        if verbose:
            print('\n🔍 STEP 5: Loading and Validating Results')
            print('-' * 30)
        
        datasets = load_ensemble_results(saved_file, verbose)
        validate_datasets(datasets, verbose)
        results['datasets'] = datasets
        
        # =================================================================
        # STEP 6: Statistical Analysis
        # =================================================================
        if verbose:
            print('\n📊 STEP 6: Statistical Analysis')
            print('-' * 30)
        
        variables = config.get('output_variables')
        
        # Calculate ensemble statistics
        ensemble_stats = calculate_ensemble_statistics(
            datasets['prediction'], variables, verbose
        )
        results['statistics'] = ensemble_stats
        
        # Calculate metrics vs truth if available
        metrics = None
        if 'truth' in datasets:
            metrics = calculate_metrics_vs_truth(
                datasets['prediction'], datasets['truth'], variables, verbose
            )
            results['metrics'] = metrics
        
        # Calculate spatial statistics
        spatial_stats = calculate_spatial_statistics(
            datasets['prediction'], variables, verbose
        )
        
        # Print summary
        print_summary_statistics(ensemble_stats, metrics, verbose)
        
        # =================================================================
        # STEP 7: Visualization
        # =================================================================
        if config.get('enable_plots', True):
            if verbose:
                print('\n🎨 STEP 7: Creating Visualizations')
                print('-' * 30)
            
            visualization_files = create_comprehensive_visualization(
                datasets['prediction'],
                datasets.get('truth'),
                ensemble_stats,
                metrics,
                variables,
                time_idx=0,
                output_dir=plot_dir,
                verbose=verbose
            )
            results['visualization_files'] = visualization_files
            
            # Create summary report
            summary_report = create_summary_report(
                datasets['prediction'],
                datasets.get('truth'),
                ensemble_stats,
                metrics,
                config.config,
                plot_dir,
                verbose
            )
        
        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        if verbose:
            total_elapsed = time.time() - start_time
            print('\n' + '=' * 60)
            print('🎉 WORKFLOW COMPLETE!')
            print('=' * 60)
            print(f'⏱️  Total time: {total_elapsed:.1f}s')
            print(f'📁 Output file: {saved_file}')
            print(f'🎨 Plots directory: {plot_dir}')
            
            if visualization_files:
                total_plots = sum(len(files) for files in visualization_files.values())
                print(f'📊 Total plots created: {total_plots}')
            
            print('\n📋 Quick Access:')
            print(f'   • Load results: xr.open_dataset("{saved_file}", group="prediction")')
            if 'truth' in datasets:
                print(f'   • Ground truth available for validation')
            print(f'   • Ensemble statistics computed for {len(variables)} variable(s)')
            print('=' * 60)
    
    except Exception as e:
        if verbose:
            print(f'\n❌ WORKFLOW FAILED: {e}')
            import traceback
            traceback.print_exc()
        raise
    
    return results


def run_analysis_only_workflow(netcdf_file: str, config: Optional[CorrDiffConfig] = None) -> Dict[str, Any]:
    """
    Run analysis and visualization on existing NetCDF results.
    
    Args:
        netcdf_file: Path to existing NetCDF results file
        config: Optional configuration (will create default if None)
        
    Returns:
        Dictionary with analysis results
    """
    if config is None:
        config = CorrDiffConfig()
    
    verbose = config.get('verbose', True)
    
    if verbose:
        print('🔍 Starting Analysis-Only Workflow')
        print('=' * 60)
        print(f'📁 Input file: {netcdf_file}')
    
    # Load datasets
    datasets = load_ensemble_results(netcdf_file, verbose)
    validate_datasets(datasets, verbose)
    
    # Get variables from the dataset
    variables = list(datasets['prediction'].data_vars.keys())
    
    # Statistical analysis
    ensemble_stats = calculate_ensemble_statistics(
        datasets['prediction'], variables, verbose
    )
    
    metrics = None
    if 'truth' in datasets:
        metrics = calculate_metrics_vs_truth(
            datasets['prediction'], datasets['truth'], variables, verbose
        )
    
    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(netcdf_file), 'analysis_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualization
    if config.get('enable_plots', True):
        visualization_files = create_comprehensive_visualization(
            datasets['prediction'],
            datasets.get('truth'),
            ensemble_stats,
            metrics,
            variables,
            time_idx=0,
            output_dir=output_dir,
            verbose=verbose
        )
        
        # Create summary report
        summary_report = create_summary_report(
            datasets['prediction'],
            datasets.get('truth'),
            ensemble_stats,
            metrics,
            config.config,
            output_dir,
            verbose
        )
    
    if verbose:
        print('\n✅ Analysis complete!')
        print(f'📊 Plots saved to: {output_dir}')
    
    return {
        'datasets': datasets,
        'statistics': ensemble_stats,
        'metrics': metrics,
        'visualization_files': visualization_files if config.get('enable_plots') else None,
        'output_dir': output_dir
    }


def quick_ensemble_inference(
    variables: str = 'Fog_index',
    num_ensembles: int = 8,
    sampling_mode: str = 'stochastic',
    inference_times: List[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Quick setup and run of ensemble inference with minimal configuration.
    
    Args:
        variables: Target variable name
        num_ensembles: Number of ensemble members
        sampling_mode: Sampling mode ('stochastic' or 'deterministic')
        inference_times: List of time strings (uses default if None)
        verbose: Enable verbose output
        
    Returns:
        Results dictionary
    """
    if inference_times is None:
        inference_times = ['2024-05-01T00:00:00', '2024-05-01T06:00:00']
    
    # Create quick configuration
    config = CorrDiffConfig({
        'variables': variables,
        'num_ensembles': num_ensembles,
        'sampling_mode': sampling_mode,
        'inference_times': inference_times,
        'verbose': verbose
    })
    
    if verbose:
        print(f'🚀 Quick Ensemble Inference: {variables}')
        print(f'⚡ {num_ensembles} members, {sampling_mode} sampling')
    
    return run_complete_corrdiff_workflow(config)


def batch_variable_analysis(
    variables_list: List[str],
    base_config: Optional[Dict] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run ensemble inference for multiple variables in batch.
    
    Args:
        variables_list: List of variable names to process
        base_config: Base configuration dictionary
        verbose: Enable verbose output
        
    Returns:
        Dictionary of results by variable
    """
    if verbose:
        print(f'📦 Batch Variable Analysis: {len(variables_list)} variables')
    
    batch_results = {}
    
    for i, var_name in enumerate(variables_list):
        if verbose:
            print(f'\n🔄 Processing variable {i+1}/{len(variables_list)}: {var_name}')
        
        # Create configuration for this variable
        var_config = CorrDiffConfig(base_config or {})
        var_config.set('variables', var_name)
        var_config.set('output_variables', [var_name])
        
        try:
            results = run_complete_corrdiff_workflow(var_config)
            batch_results[var_name] = results
            
            if verbose:
                print(f'✅ {var_name} completed successfully')
        
        except Exception as e:
            if verbose:
                print(f'❌ {var_name} failed: {e}')
            batch_results[var_name] = {'error': str(e)}
    
    if verbose:
        successful = sum(1 for r in batch_results.values() if 'error' not in r)
        print(f'\n📊 Batch complete: {successful}/{len(variables_list)} successful')
    
    return batch_results