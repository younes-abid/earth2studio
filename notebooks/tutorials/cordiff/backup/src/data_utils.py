"""
Data handling and I/O utilities for CorrDiff experiments.
"""
import os
import sys
import time
import torch
import numpy as np
import xarray as xr
import netCDF4 as nc
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from earth2studio.data import prep_data_array
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array

# Add the current directory to Python path to find custom_data_loader
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import the custom data loader
from custom_data_loader import CustomCorrDiffDataSource


def load_data_source(config, verbose: bool = True):
    """Load and initialize data source from configuration."""
    start_time = time.time() if verbose else None
    
    if verbose:
        print('📂 Loading data source...')
    
    # Get data paths
    data_file, stats_file = config.get_data_paths()
    
    if verbose:
        print(f'   Data file: {Path(data_file).name}')
        print(f'   Stats file: {Path(stats_file).name}')
    
    try:
        data_source = CustomCorrDiffDataSource(
            data_paths=[data_file],
            stats_path=stats_file,
            input_variables=config.get('input_variables'),
            output_variables=config.get('output_variables'),
            cache_data=True
        )
        
        if verbose:
            elapsed = time.time() - start_time
            print(f'✅ Data source loaded in {elapsed:.3f}s')
            print(f'   Available samples: {getattr(data_source, "total_samples", "N/A")}')
            print(f'   Input shape: {getattr(data_source, "input_shape", "N/A")}')
        
        return data_source
        
    except Exception as e:
        if verbose:
            print(f'❌ Error loading custom data source: {e}')
            print(f'   Data file exists: {os.path.exists(data_file)}')
            print(f'   Stats file exists: {os.path.exists(stats_file)}')
        raise RuntimeError(f"Failed to load data source: {e}")


def run_ensemble_inference(times: List[str], model, data_source, device, verbose: bool = True) -> Dict:
    """
    Run ensemble inference workflow.
    
    Args:
        times: List of time strings to process
        model: Ensemble CorrDiff model
        data_source: Data source instance
        device: PyTorch device
        verbose: Enable verbose output
        
    Returns:
        Dictionary with inference results
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print(f'🚀 Starting ensemble inference on {device}')
        print(f'📅 Processing {len(times)} time steps')
        print(f'🎯 Generating {model.number_of_samples} ensemble members per time step')
    
    # Move model to device
    model = model.to(device)
    
    # Storage for all results
    all_predictions = []
    all_inputs = []
    all_truth = []  # For ground truth if available
    all_coords = []
    processed_times = []
    
    # Process each time step
    for i, time_step in enumerate(times):
        step_start = time.time() if verbose else None
        
        if verbose:
            print(f'🔄 Processing time {i+1}/{len(times)}: {time_step}')
        
        try:
            # Load data for this time step
            time_array = to_time_array([time_step])
            
            # Get data from data source (this returns an xarray.DataArray now)
            data_array = data_source(time_array, model.input_variables)
            
            if verbose:
                print(f'   📊 Raw data shape: {data_array.shape}')
                print(f'   📊 Data type: {type(data_array)}')
            
            # Prepare data array for model input
            x, coords = prep_data_array(data_array, device=device)
            x, coords = map_coords(x, coords, model.input_coords())
            
            if verbose:
                print(f'   📊 Prepared input shape: {x.shape}')
            
            # Store input for later saving
            all_inputs.append(x.cpu())
            
            # Try to get ground truth if available
            try:
                truth_data = data_source(time_array, model.output_variables)
                if truth_data is not None and truth_data.size > 0:
                    truth_tensor, _ = prep_data_array(truth_data, device=device)
                    all_truth.append(truth_tensor.cpu())
                    if verbose:
                        print(f'   ✅ Ground truth loaded: {truth_tensor.shape}')
            except Exception as truth_error:
                if verbose and i == 0:  # Only print once
                    print(f'   ⚠️  No ground truth available: {truth_error}')
            
            # Run ensemble inference
            with torch.no_grad():
                pred, pred_coords = model(x, coords)
            
            if verbose:
                step_elapsed = time.time() - step_start
                print(f'   ✅ Generated ensemble shape: {pred.shape} in {step_elapsed:.3f}s')
            
            # Store results
            all_predictions.append(pred.cpu())
            all_coords.append(pred_coords)
            processed_times.append(time_array[0])
            
        except Exception as e:
            if verbose:
                print(f'   ❌ Error processing {time_step}: {e}')
                import traceback
                traceback.print_exc()
            continue
    
    if verbose:
        total_elapsed = time.time() - start_time
        print(f'✅ Ensemble inference complete in {total_elapsed:.3f}s')
        print(f'   Processed {len(processed_times)} time steps')
    
    return {
        'predictions': all_predictions,
        'inputs': all_inputs,
        'truth': all_truth if all_truth else None,
        'coords': all_coords,
        'times': processed_times
    }


def save_ensemble_netcdf(results: Dict, config, output_path: str, verbose: bool = True) -> str:
    """
    Save ensemble results in PhysicsNeMo-style NetCDF format.
    
    Args:
        results: Results dictionary from inference
        config: Configuration object
        output_path: Output file path
        verbose: Enable verbose output
        
    Returns:
        Path to saved file
    """
    start_time = time.time() if verbose else None
    
    if verbose:
        print(f'💾 Saving ensemble results to: {output_path}')
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract dimensions and variables
    n_times = len(results['times'])
    n_ensembles = config.get('num_ensembles')
    output_variables = config.get('output_variables')
    input_variables = config.get('input_variables')
    
    # Get grid information from first prediction
    if results['predictions']:
        pred_shape = results['predictions'][0].shape
        n_lat, n_lon = pred_shape[-2], pred_shape[-1]
        
        # Try to get input grid info
        if results['inputs']:
            input_shape = results['inputs'][0].shape
            n_input_lat, n_input_lon = input_shape[-2], input_shape[-1]
        else:
            n_input_lat, n_input_lon = n_lat, n_lon
    else:
        raise ValueError("No predictions to save")
    
    if verbose:
        print(f'   📐 Dimensions: time={n_times}, ensemble={n_ensembles}, lat={n_lat}, lon={n_lon}')
        has_truth = results.get('truth') is not None
        print(f'   🎯 Ground truth available: {has_truth}')
    
    # Create NetCDF file
    with nc.Dataset(output_path, 'w', format='NETCDF4') as f:
        
        # Global attributes
        f.title = 'CorrDiff Ensemble Predictions'
        f.description = f'Ensemble forecasts with {n_ensembles} members using {config.get("sampling_mode")} sampling'
        f.created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.sampling_mode = str(config.get('sampling_mode'))
        f.num_steps = config.get('number_of_steps')
        f.solver = str(config.get('solver'))
        f.hr_mean_conditioning = str(config.get('hr_mean_conditioning'))
        f.base_seed = config.get('seed_base')
        f.variables = str(output_variables)
        
        # =================================================================
        # PREDICTION GROUP - Main ensemble predictions
        # =================================================================
        pred_group = f.createGroup('prediction')
        
        # Dimensions
        pred_group.createDimension('time', n_times)
        pred_group.createDimension('ensemble', n_ensembles)
        pred_group.createDimension('y', n_lat)
        pred_group.createDimension('x', n_lon)
        
        # Coordinate variables
        time_var = pred_group.createVariable('time', 'f8', ('time',))
        ens_var = pred_group.createVariable('ensemble', 'i4', ('ensemble',))
        y_var = pred_group.createVariable('y', 'f4', ('y',))
        x_var = pred_group.createVariable('x', 'f4', ('x',))
        
        # Set coordinate values
        time_var[:] = [(t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'h') 
                       for t in results['times']]
        time_var.units = 'hours since 1970-01-01 00:00:00'
        ens_var[:] = np.arange(n_ensembles)
        y_var[:] = np.arange(n_lat)
        x_var[:] = np.arange(n_lon)
        
        # Data variables for each output variable
        for var_idx, var_name in enumerate(output_variables):
            var = pred_group.createVariable(var_name, 'f4', ('ensemble', 'time', 'y', 'x'), 
                                          compression='zlib', complevel=4)
            
            # Collect data for this variable across all times
            var_data = np.zeros((n_ensembles, n_times, n_lat, n_lon))
            for t_idx, pred in enumerate(results['predictions']):
                var_data[:, t_idx, :, :] = pred[0, :, var_idx, :, :]  # [batch, sample, variable, lat, lon]
            
            var[:] = var_data
            var.long_name = f'Ensemble predictions for {var_name}'
            var.units = 'model_units'
        
        # =================================================================
        # INPUT GROUP - Low resolution inputs
        # =================================================================
        input_group = f.createGroup('input')
        
        # Dimensions
        input_group.createDimension('time', n_times)
        input_group.createDimension('y_input', n_input_lat)
        input_group.createDimension('x_input', n_input_lon)
        
        # Coordinate variables
        time_input = input_group.createVariable('time', 'f8', ('time',))
        y_input = input_group.createVariable('y', 'f4', ('y_input',))
        x_input = input_group.createVariable('x', 'f4', ('x_input',))
        
        time_input[:] = time_var[:]
        time_input.units = time_var.units
        y_input[:] = np.arange(n_input_lat)
        x_input[:] = np.arange(n_input_lon)
        
        # Data variables for each input variable
        for var_idx, var_name in enumerate(input_variables):
            var = input_group.createVariable(var_name, 'f4', ('time', 'y_input', 'x_input'),
                                            compression='zlib', complevel=4)
            
            # Collect input data across all times
            if results['inputs']:
                var_data = np.zeros((n_times, n_input_lat, n_input_lon))
                for t_idx, inp in enumerate(results['inputs']):
                    if var_idx < inp.shape[1]:  # Check if variable exists
                        var_data[t_idx, :, :] = inp[0, var_idx, :, :]  # [batch, variable, lat, lon]
                
                var[:] = var_data
            
            var.long_name = f'Input {var_name}'
            var.units = 'model_units'
        
        # =================================================================
        # TRUTH GROUP - Ground truth (if available)
        # =================================================================
        truth_group = f.createGroup('truth')
        
        if results.get('truth') is not None:
            if verbose:
                print('   📋 Saving ground truth data')
            
            # Dimensions
            truth_group.createDimension('time', n_times)
            truth_group.createDimension('y', n_lat)
            truth_group.createDimension('x', n_lon)
            
            # Coordinate variables
            time_truth = truth_group.createVariable('time', 'f8', ('time',))
            y_truth = truth_group.createVariable('y', 'f4', ('y',))
            x_truth = truth_group.createVariable('x', 'f4', ('x',))
            
            time_truth[:] = time_var[:]
            time_truth.units = time_var.units
            y_truth[:] = y_var[:]
            x_truth[:] = x_var[:]
            
            # Data variables for each output variable
            for var_idx, var_name in enumerate(output_variables):
                var = truth_group.createVariable(var_name, 'f4', ('time', 'y', 'x'),
                                                compression='zlib', complevel=4)
                
                # Collect truth data across all times
                var_data = np.zeros((n_times, n_lat, n_lon))
                for t_idx, truth in enumerate(results['truth']):
                    if var_idx < truth.shape[1]:  # Check if variable exists
                        var_data[t_idx, :, :] = truth[0, var_idx, :, :]  # [batch, variable, lat, lon]
                
                var[:] = var_data
                var.long_name = f'Ground truth {var_name}'
                var.units = 'model_units'
        else:
            truth_group.description = 'Ground truth data not available in current setup'
            if verbose:
                print('   ⚠️  No ground truth data to save')
    
    if verbose:
        elapsed = time.time() - start_time
        print(f'✅ Ensemble results saved in {elapsed:.3f}s')
    
    return output_path


def load_ensemble_results(netcdf_path: str, verbose: bool = True) -> Dict[str, xr.Dataset]:
    """
    Load ensemble results from NetCDF file.
    
    Args:
        netcdf_path: Path to NetCDF file
        verbose: Enable verbose output
        
    Returns:
        Dictionary with datasets by group
    """
    if verbose:
        print(f'📂 Loading ensemble results from: {Path(netcdf_path).name}')
    
    datasets = {}
    
    try:
        # Load prediction data
        datasets['prediction'] = xr.open_dataset(netcdf_path, group='prediction')
        
        # Load input data
        datasets['input'] = xr.open_dataset(netcdf_path, group='input')
        
        # Try to load truth data
        try:
            datasets['truth'] = xr.open_dataset(netcdf_path, group='truth')
            has_truth = True
        except:
            has_truth = False
        
        if verbose:
            pred_ds = datasets['prediction']
            print(f'   📊 Prediction dataset: {dict(pred_ds.dims)}')
            print(f'   📋 Variables: {list(pred_ds.data_vars.keys())}')
            print(f'   🎯 Ground truth available: {has_truth}')
        
        return datasets
        
    except Exception as e:
        if verbose:
            print(f'❌ Error loading ensemble results: {e}')
        raise


def validate_datasets(datasets: Dict[str, xr.Dataset], verbose: bool = True) -> bool:
    """
    Validate loaded datasets for consistency.
    
    Args:
        datasets: Dictionary of datasets
        verbose: Enable verbose output
        
    Returns:
        True if validation passes
    """
    if verbose:
        print('🔍 Validating datasets...')
    
    issues = []
    
    # Check if prediction dataset exists
    if 'prediction' not in datasets:
        issues.append("Missing prediction dataset")
    else:
        pred_ds = datasets['prediction']
        if 'ensemble' not in pred_ds.dims:
            issues.append("Prediction dataset missing ensemble dimension")
    
    # Check time consistency
    if 'prediction' in datasets and 'input' in datasets:
        pred_times = datasets['prediction'].time.values
        input_times = datasets['input'].time.values
        
        if len(pred_times) != len(input_times):
            issues.append("Prediction and input have different number of time steps")
    
    # Check truth consistency if available
    if 'truth' in datasets and 'prediction' in datasets:
        truth_times = datasets['truth'].time.values
        pred_times = datasets['prediction'].time.values
        
        if len(truth_times) != len(pred_times):
            issues.append("Truth and prediction have different number of time steps")
    
    if issues:
        if verbose:
            print("❌ Validation failed:")
            for issue in issues:
                print(f"   - {issue}")
        return False
    else:
        if verbose:
            print("✅ Dataset validation passed")
        return True