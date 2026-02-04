"""
Simple utilities for inference and result saving.
"""
import os
import time
import torch
import numpy as np
import xarray as xr
import netCDF4 as nc
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from earth2studio.data import prep_data_array
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array


def run_ensemble_inference(times: List[str], model, data_source, device, verbose: bool = True) -> Dict:
    """Run ensemble inference for given time steps - FIXED to match working version."""
    start_time = time.time() if verbose else None
    
    if verbose:
        print(f'🚀 Starting ensemble inference on {device}')
        print(f'📅 Processing {len(times)} time steps')
        print(f'🎯 Generating {model.num_ensembles} ensemble members per time step')
    
    model = model.to(device)
    
    # Storage
    predictions = []
    inputs = []
    ground_truth = []
    processed_times = []
    
    # Check if ground truth is available
    has_ground_truth = data_source.has_ground_truth()
    if verbose:
        print(f'🎯 Ground truth available: {has_ground_truth}')
    
    # Process each time step
    for i, time_step in enumerate(times):
        if verbose:
            print(f'🔄 Processing time {i+1}/{len(times)}: {time_step}')
        
        try:
            # Load input data - FIXED: handle DataArray properly
            time_array = to_time_array([time_step])
            data_array = data_source(time_array, model.input_variables)
            
            if verbose:
                print(f'✅ Loaded DataArray with shape: {data_array.shape}')
                print(f'   Dimensions: {dict(data_array.sizes)}')
                print(f'   📊 Input shape: {data_array.shape}')
            
            # FIXED: Prepare data for model using Earth2Studio tools
            x, coords = prep_data_array(data_array, device=device)
            x, coords = map_coords(x, coords, model.input_coords())
            
            # Store input
            inputs.append(x.cpu())
            
            # Load ground truth if available
            if has_ground_truth:
                try:
                    # Load ground truth from output group
                    truth_data = data_source(time_array, model.output_variables)
                    truth_tensor, _ = prep_data_array(truth_data, device=device)
                    
                    if verbose and i == 0:  # Only print once
                        print(f'   ✅ Ground truth shape: {truth_tensor.shape}')
                    
                    # FIXED: Store ground truth in correct format
                    ground_truth.append(truth_tensor.cpu())
                    
                except Exception as e:
                    if verbose:
                        print(f'   ⚠️  Ground truth loading failed: {e}')
                    has_ground_truth = False
            
            # FIXED: Run inference with proper error handling
            with torch.no_grad():
                pred, pred_coords = model(x, coords)
            
            if verbose:
                print(f'   ✅ Generated ensemble: {pred.shape}')
            
            # Store prediction
            predictions.append(pred.cpu())
            processed_times.append(time_array[0])
            
            # Memory cleanup
            del x, pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            if verbose:
                print(f'   ❌ Error processing {time_step}: {repr(e)}')
            continue
    
    if verbose:
        total_elapsed = time.time() - start_time
        print(f'✅ Ensemble inference complete in {total_elapsed:.3f}s')
        print(f'   Processed {len(processed_times)} time steps')
    
    return {
        'predictions': predictions,
        'inputs': inputs,
        'ground_truth': ground_truth if ground_truth else None,
        'times': processed_times,
        'has_ground_truth': has_ground_truth and len(ground_truth) > 0
    }


def save_ensemble_netcdf(results: Dict, model, config: Dict, output_path: str, verbose: bool = True) -> str:
    """Save ensemble results to NetCDF file - FIXED to handle grid coordinates properly."""
    start_time = time.time() if verbose else None
    
    if verbose:
        print(f'💾 Saving ensemble results to: {output_path}')
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract dimensions
    n_times = len(results['times'])
    n_ensembles = config['num_ensembles']
    output_variables = config['output_variables']
    input_variables = config['input_variables']
    
    # FIXED: Get grid dimensions from model attributes
    n_lat = len(model.output_lat)
    n_lon = len(model.output_lon)
    n_input_lat = len(model.input_lat)
    n_input_lon = len(model.input_lon)
    
    if verbose:
        print(f'   📐 Dimensions: time={n_times}, ensemble={n_ensembles}')
        print(f'   📐 Output grid: {n_lat}x{n_lon}, Input grid: {n_input_lat}x{n_input_lon}')
        print(f'   🎯 Ground truth: {results["has_ground_truth"]}')
    
    # Create NetCDF file
    with nc.Dataset(output_path, 'w', format='NETCDF4') as f:
        
        # Global attributes
        f.title = 'CorrDiff Ensemble Predictions'
        f.description = f'Ensemble forecasts with {n_ensembles} members'
        f.created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.sampling_mode = config['sampling_mode']
        f.num_steps = config['number_of_steps']
        f.solver = config['solver']
        f.hr_mean_conditioning = str(config['hr_mean_conditioning'])
        f.base_seed = config['seed_base']
        f.variables = str(output_variables)
        
        # =================================================================
        # PREDICTION GROUP
        # =================================================================
        pred_group = f.createGroup('prediction')
        pred_group.createDimension('time', n_times)
        pred_group.createDimension('ensemble', n_ensembles)
        pred_group.createDimension('y', n_lat)
        pred_group.createDimension('x', n_lon)
        
        # Coordinates
        time_var = pred_group.createVariable('time', 'f8', ('time',))
        ens_var = pred_group.createVariable('ensemble', 'i4', ('ensemble',))
        y_var = pred_group.createVariable('y', 'f4', ('y',))
        x_var = pred_group.createVariable('x', 'f4', ('x',))
        
        time_var[:] = [(t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'h') 
                       for t in results['times']]
        time_var.units = 'hours since 1970-01-01 00:00:00'
        ens_var[:] = np.arange(n_ensembles)
        y_var[:] = model.output_lat  # FIXED: Use model coordinates directly
        x_var[:] = model.output_lon
        
        # Prediction variables
        for var_idx, var_name in enumerate(output_variables):
            var = pred_group.createVariable(var_name, 'f4', ('ensemble', 'time', 'y', 'x'), 
                                          compression='zlib', complevel=4)
            
            var_data = np.zeros((n_ensembles, n_times, n_lat, n_lon))
            for t_idx, pred in enumerate(results['predictions']):
                var_data[:, t_idx, :, :] = pred[0, :, var_idx, :, :]
            
            var[:] = var_data
            var.long_name = f'Ensemble predictions for {var_name}'
        
        # =================================================================
        # INPUT GROUP
        # =================================================================
        input_group = f.createGroup('input')
        input_group.createDimension('time', n_times)
        input_group.createDimension('y_input', n_input_lat)
        input_group.createDimension('x_input', n_input_lon)
        
        time_input = input_group.createVariable('time', 'f8', ('time',))
        y_input = input_group.createVariable('y', 'f4', ('y_input',))
        x_input = input_group.createVariable('x', 'f4', ('x_input',))
        
        time_input[:] = time_var[:]
        time_input.units = time_var.units
        y_input[:] = model.input_lat  # FIXED: Use model coordinates directly
        x_input[:] = model.input_lon
        
        # Input variables
        for var_idx, var_name in enumerate(input_variables):
            var = input_group.createVariable(var_name, 'f4', ('time', 'y_input', 'x_input'),
                                            compression='zlib', complevel=4)
            
            var_data = np.zeros((n_times, n_input_lat, n_input_lon))
            for t_idx, inp in enumerate(results['inputs']):
                if var_idx < inp.shape[1]:
                    var_data[t_idx, :, :] = inp[0, var_idx, :, :]
            
            var[:] = var_data
            var.long_name = f'Input {var_name}'
        
        # =================================================================
        # GROUND TRUTH GROUP (if available)
        # =================================================================
        if results['has_ground_truth'] and results['ground_truth']:
            if verbose:
                print('   📋 Saving ground truth data')
            
            truth_group = f.createGroup('truth')
            truth_group.createDimension('time', n_times)
            truth_group.createDimension('y', n_lat)
            truth_group.createDimension('x', n_lon)
            
            time_truth = truth_group.createVariable('time', 'f8', ('time',))
            y_truth = truth_group.createVariable('y', 'f4', ('y',))
            x_truth = truth_group.createVariable('x', 'f4', ('x',))
            
            time_truth[:] = time_var[:]
            time_truth.units = time_var.units
            y_truth[:] = y_var[:]
            x_truth[:] = x_var[:]
            
            # Truth variables
            for var_idx, var_name in enumerate(output_variables):
                var = truth_group.createVariable(var_name, 'f4', ('time', 'y', 'x'),
                                                compression='zlib', complevel=4)
                
                var_data = np.zeros((n_times, n_lat, n_lon))
                for t_idx, truth in enumerate(results['ground_truth']):
                    if var_idx < truth.shape[1]:
                        var_data[t_idx, :, :] = truth[0, var_idx, :, :]
                
                var[:] = var_data
                var.long_name = f'Ground truth {var_name}'
        
        else:
            # Create empty truth group with description
            truth_group = f.createGroup('truth')
            truth_group.description = 'Ground truth data not available'
    
    if verbose:
        elapsed = time.time() - start_time
        file_size = Path(output_path).stat().st_size / 1024**2
        print(f'✅ Results saved in {elapsed:.3f}s ({file_size:.1f} MB)')
    
    return output_path


def load_and_validate_results(netcdf_path: str, verbose: bool = True) -> Dict[str, xr.Dataset]:
    """Load and validate ensemble results."""
    if verbose:
        print(f'📂 Loading results from: {Path(netcdf_path).name}')
    
    datasets = {}
    
    try:
        # Load datasets
        datasets['prediction'] = xr.open_dataset(netcdf_path, group='prediction')
        datasets['input'] = xr.open_dataset(netcdf_path, group='input')
        
        # Try to load truth
        try:
            datasets['truth'] = xr.open_dataset(netcdf_path, group='truth')
            has_truth = len(datasets['truth'].data_vars) > 0
        except:
            has_truth = False
        
        if verbose:
            pred_ds = datasets['prediction']
            print(f'   📊 Prediction dataset: {dict(pred_ds.dims)}')
            print(f'   📋 Variables: {list(pred_ds.data_vars.keys())}')
            print(f'   🎯 Ground truth: {has_truth}')
        
        return datasets
        
    except Exception as e:
        if verbose:
            print(f'❌ Error loading results: {e}')
        raise
