# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Optimized output handling for CorrDiff ensemble results.
Streamlined NetCDF saving with proper coordinate integration for visualization.
"""

import os
import numpy as np
import netCDF4 as nc
import xarray as xr
from datetime import datetime


def _load_ground_truth_data(config, requested_times):
    """Load ground truth data if available and requested"""
    if not config.SAVE_GROUND_TRUTH:
        return None
    
    try:
        with nc.Dataset(config.DATA_FILE, 'r') as ds:
            if "output" not in ds.groups:
                return None
        
        with xr.open_dataset(config.DATA_FILE, group="output") as output_ds:
            truth_data = {}
            for var_name in config.OUTPUT_VARIABLES:
                if var_name in output_ds.data_vars:
                    var_data = output_ds[var_name][:len(requested_times)]
                    truth_data[var_name] = np.array(var_data)
            return truth_data if truth_data else None
    except:
        return None


def _create_coordinates(group, ensemble_model, n_times, results, include_ensemble=False):
    """Create coordinate variables with proper metadata for visualization"""
    # Time coordinate
    time_var = group.createVariable('time', 'f8', ('time',))
    time_var[:] = [(t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'h') 
                   for t in results['times']]
    time_var.units = 'hours since 1970-01-01 00:00:00'
    time_var.long_name = 'time'
    time_var.standard_name = 'time'
    
    # Ensemble coordinate (if needed)
    if include_ensemble:
        ens_var = group.createVariable('ensemble', 'i4', ('ensemble',))
        ens_var[:] = np.arange(ensemble_model.number_of_samples)
        ens_var.long_name = 'ensemble member'
    
    # Spatial coordinates - ALWAYS create as proper coordinate dimensions
    if hasattr(ensemble_model, 'output_lat_2d') and ensemble_model.output_lat_2d is not None:
        # 2D coordinates for visualization - save as coordinate variables
        lat_var = group.createVariable('lat', 'f4', ('y', 'x'))
        lon_var = group.createVariable('lon', 'f4', ('y', 'x'))
        lat_var[:] = ensemble_model.output_lat_2d
        lon_var[:] = ensemble_model.output_lon_2d
        
        # Mark as coordinates (this is the key!)
        lat_var.coordinates = "lat lon"
        lon_var.coordinates = "lat lon"
    else:
        # 1D coordinates as coordinate variables
        lat_var = group.createVariable('lat', 'f4', ('y',))
        lon_var = group.createVariable('lon', 'f4', ('x',))
        lat_var[:] = ensemble_model.output_lat
        lon_var[:] = ensemble_model.output_lon
        
        # Mark as coordinates
        lat_var.coordinates = "lat lon"
        lon_var.coordinates = "lat lon"
    
    # Add proper coordinate metadata
    lat_var.long_name = 'latitude'
    lon_var.long_name = 'longitude'
    lat_var.units = 'degrees_north'
    lon_var.units = 'degrees_east'
    lat_var.standard_name = 'latitude'
    lon_var.standard_name = 'longitude'


def save_ensemble_netcdf(config, results, ensemble_model, output_path=None):
    """
    Save ensemble results with proper coordinate integration for visualization.
    
    Returns path to saved file.
    """
    output_path = output_path or config.OUTPUT_ENSEMBLE_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f'💾 Saving to: {os.path.basename(output_path)}')
    
    # Extract dimensions
    n_times = len(results['times'])
    n_ensembles = config.NUM_ENSEMBLES
    n_lat = len(ensemble_model.output_lat)
    n_lon = len(ensemble_model.output_lon)
    
    with nc.Dataset(output_path, 'w', format='NETCDF4') as f:
        # Global attributes
        f.title = f'CorrDiff Ensemble Predictions - {config.VARIABLES}'
        f.description = f'Ensemble forecasts with {n_ensembles} members'
        f.created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.sampling_mode = config.SAMPLING_MODE
        f.num_steps = config.NUMBER_OF_STEPS
        f.base_seed = config.SEED_BASE
        
        # =================================================================
        # PREDICTION GROUP - Main ensemble predictions
        # =================================================================
        if config.SAVE_PREDICTIONS:
            pred_group = f.createGroup('prediction')
            
            # Dimensions
            pred_group.createDimension('time', n_times)
            pred_group.createDimension('ensemble', n_ensembles)
            pred_group.createDimension('y', n_lat)
            pred_group.createDimension('x', n_lon)
            
            # Coordinates
            _create_coordinates(pred_group, ensemble_model, n_times, results, include_ensemble=True)
            
            # Data variables
            for var_idx, var_name in enumerate(config.OUTPUT_VARIABLES):
                var = pred_group.createVariable(
                    var_name, 'f4', ('ensemble', 'time', 'y', 'x'), 
                    compression='zlib', complevel=4
                )
                
                # Collect data for this variable
                var_data = np.zeros((n_ensembles, n_times, n_lat, n_lon))
                for t_idx, pred in enumerate(results['predictions']):
                    var_data[:, t_idx, :, :] = pred[0, :, var_idx, :, :]
                
                var[:] = var_data
                var.long_name = f'{var_name} ensemble predictions'
                var.units = 'model_units'
        
        # =================================================================
        # INPUT GROUP - Low resolution inputs
        # =================================================================
        if config.SAVE_INPUT_VARIABLES:
            input_group = f.createGroup('input')
            
            # Dimensions
            input_group.createDimension('time', n_times)
            input_group.createDimension('y', len(ensemble_model.input_lat))
            input_group.createDimension('x', len(ensemble_model.input_lon))
            
            # Coordinates - Use WRF-consistent coordinate ranges for INPUT group
            time_input = input_group.createVariable('time', 'f8', ('time',))
            lat_input = input_group.createVariable('lat', 'f4', ('y',))
            lon_input = input_group.createVariable('lon', 'f4', ('x',))
            
            time_input[:] = [(t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'h') 
                            for t in results['times']]
            time_input.units = 'hours since 1970-01-01 00:00:00'
            
            # Create INPUT coordinates that match the WRF coordinate region
            if hasattr(ensemble_model, 'output_lat_2d') and ensemble_model.output_lat_2d is not None:
                # Use coordinate ranges from WRF data for INPUT group
                lat_min = float(ensemble_model.output_lat_2d.min())
                lat_max = float(ensemble_model.output_lat_2d.max())
                lon_min = float(ensemble_model.output_lon_2d.min())
                lon_max = float(ensemble_model.output_lon_2d.max())
                
                # Create 1D coordinate arrays that span the same geographic region
                input_lat_coords = np.linspace(lat_min, lat_max, len(ensemble_model.input_lat))
                input_lon_coords = np.linspace(lon_min, lon_max, len(ensemble_model.input_lon))
            else:
                # Fallback to dummy coordinates if no WRF coords available
                input_lat_coords = ensemble_model.input_lat
                input_lon_coords = ensemble_model.input_lon
            
            lat_input[:] = input_lat_coords
            lon_input[:] = input_lon_coords
            lat_input.units = 'degrees_north'
            lon_input.units = 'degrees_east'
            
            # Data variables
            for var_idx, var_name in enumerate(config.INPUT_VARIABLES):
                var = input_group.createVariable(
                    var_name, 'f4', ('time', 'y', 'x'),
                    compression='zlib', complevel=4
                )
                
                # Collect input data
                var_data = np.zeros((n_times, len(ensemble_model.input_lat), len(ensemble_model.input_lon)))
                for t_idx, inp in enumerate(results['inputs']):
                    var_data[t_idx, :, :] = inp[0, var_idx, :, :]
                
                var[:] = var_data
                var.long_name = f'{var_name} input'
                var.units = 'model_units'
        
        # =================================================================
        # TRUTH GROUP - Ground truth (conditional)
        # =================================================================
        truth_data = _load_ground_truth_data(config, results['times'])
        if truth_data:
            truth_group = f.createGroup('truth')
            truth_group.createDimension('time', n_times)
            truth_group.createDimension('y', n_lat)
            truth_group.createDimension('x', n_lon)
            
            # Coordinates
            _create_coordinates(truth_group, ensemble_model, n_times, results, include_ensemble=False)
            
            # Variables
            for var_name in config.OUTPUT_VARIABLES:
                if var_name in truth_data:
                    truth_var = truth_group.createVariable(
                        var_name, 'f4', ('time', 'y', 'x'), 
                        compression='zlib', complevel=4
                    )
                    truth_var[:] = truth_data[var_name]
                    truth_var.long_name = f'{var_name} ground truth'
                    truth_var.units = 'model_units'
    
    print(f'✅ Saved: {n_times} times, {n_ensembles} members')
    return output_path


def verify_output(output_path):
    """Verify saved NetCDF file and print summary"""
    print(f'📁 Verifying: {os.path.basename(output_path)}')
    
    try:
        # Check each group
        for group_name in ['prediction', 'input', 'truth']:
            try:
                with xr.open_dataset(output_path, group=group_name) as ds:
                    print(f'   ✅ {group_name}: {dict(ds.dims)} - {list(ds.data_vars.keys())}')
            except (OSError, KeyError):
                if group_name != 'truth':  # Truth is optional
                    print(f'   ❌ Missing {group_name} group')
    
    except Exception as e:
        print(f'❌ Verification error: {e}')