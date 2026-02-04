# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Output handling for CorrDiff ensemble results.
Contains functions to save results in PhysicsNeMo-compatible NetCDF format.
"""

import os
import numpy as np
import netCDF4 as nc
import xarray as xr
from datetime import datetime


def save_ensemble_netcdf(config, results, ensemble_model, output_path=None):
    """
    Save ensemble results in PhysicsNeMo-style NetCDF format.
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object
    results : dict
        Results dictionary from run_ensemble_inference
    ensemble_model : EnsembleFogIndexCorrDiff
        Ensemble model (for coordinate information)
    output_path : str, optional
        Output file path (uses config.OUTPUT_ENSEMBLE_FILE if None)
    
    Returns
    -------
    str
        Path to saved file
    """
    output_path = output_path or config.OUTPUT_ENSEMBLE_FILE
    
    print(f'💾 Saving ensemble results to: {output_path}')
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create NetCDF file
    with nc.Dataset(output_path, 'w', format='NETCDF4') as f:
        
        # Global attributes
        f.title = 'CorrDiff Ensemble Predictions'
        f.description = f'Ensemble forecasts with {config.NUM_ENSEMBLES} members using {config.SAMPLING_MODE} sampling'
        f.created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.sampling_mode = config.SAMPLING_MODE
        f.num_steps = config.NUMBER_OF_STEPS
        f.solver = config.SOLVER
        f.hr_mean_conditioning = str(config.HR_MEAN_CONDITIONING)
        f.base_seed = config.SEED_BASE
        
        # Extract dimensions
        n_times = len(results['times'])
        n_ensembles = config.NUM_ENSEMBLES
        n_lat = len(ensemble_model.output_lat)
        n_lon = len(ensemble_model.output_lon)
        
        print(f'   📐 Dimensions: time={n_times}, ensemble={n_ensembles}, lat={n_lat}, lon={n_lon}')
        
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
        time_var[:] = [(t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'h') for t in results['times']]
        time_var.units = 'hours since 1970-01-01 00:00:00'
        ens_var[:] = np.arange(n_ensembles)
        y_var[:] = ensemble_model.output_lat
        x_var[:] = ensemble_model.output_lon
        
        # Data variables for each output variable
        for var_idx, var_name in enumerate(config.OUTPUT_VARIABLES):
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
        input_group.createDimension('y_input', len(ensemble_model.input_lat))
        input_group.createDimension('x_input', len(ensemble_model.input_lon))
        
        # Coordinate variables
        time_input = input_group.createVariable('time', 'f8', ('time',))
        y_input = input_group.createVariable('y', 'f4', ('y_input',))
        x_input = input_group.createVariable('x', 'f4', ('x_input',))
        
        time_input[:] = time_var[:]
        time_input.units = time_var.units
        y_input[:] = ensemble_model.input_lat
        x_input[:] = ensemble_model.input_lon
        
        # Data variables for each input variable
        for var_idx, var_name in enumerate(config.INPUT_VARIABLES):
            var = input_group.createVariable(var_name, 'f4', ('time', 'y_input', 'x_input'),
                                            compression='zlib', complevel=4)
            
            # Collect input data across all times
            var_data = np.zeros((n_times, len(ensemble_model.input_lat), len(ensemble_model.input_lon)))
            for t_idx, inp in enumerate(results['inputs']):
                var_data[t_idx, :, :] = inp[0, var_idx, :, :]  # [batch, variable, lat, lon]
            
            var[:] = var_data
            var.long_name = f'Input {var_name}'
            var.units = 'model_units'
        
        # =================================================================
        # TRUTH GROUP - Ground truth (placeholder for future extension)
        # =================================================================
        truth_group = f.createGroup('truth')
        truth_group.description = 'Ground truth data (not available in current setup)'
    
    print(f'✅ Ensemble results saved to {output_path}')
    return output_path


def verify_output(output_path):
    """
    Verify the saved NetCDF file and print summary information.
    
    Parameters
    ----------
    output_path : str
        Path to saved NetCDF file
    """
    print(f'\n📁 Verifying saved file:')
    
    try:
        with xr.open_dataset(output_path, group='prediction') as ds:
            print(f'   📊 Prediction dataset:')
            print(f'   📐 Dimensions: {dict(ds.dims)}')
            print(f'   📋 Variables: {list(ds.data_vars.keys())}')
        
        with xr.open_dataset(output_path, group='input') as ds:
            print(f'   📊 Input dataset:')
            print(f'   📐 Dimensions: {dict(ds.dims)}')
            print(f'   📋 Variables: {list(ds.data_vars.keys())}')
            
    except Exception as e:
        print(f'   ❌ Error verifying file: {e}')