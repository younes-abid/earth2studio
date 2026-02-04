"""
Simple data loader for CorrDiff experiments.
"""
import os
import json
import torch
import numpy as np
import xarray as xr
from typing import List, Optional, Dict, Any
from earth2studio.utils.time import to_time_array


class SimpleDataSource:
    """Simplified data source for CorrDiff experiments - FIXED to match working 1.1 version."""
    
    def __init__(self, data_file: str, stats_file: str, 
                 input_variables: List[str], output_variables: List[str],
                 verbose: bool = False):
        self.data_file = data_file
        self.stats_file = stats_file
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.verbose = verbose
        
        if verbose:
            print(f"📂 Loading data from: {os.path.basename(data_file)}")
            
        # Load dataset and analyze structure
        self._analyze_data_structure()
        
        # Load stats for normalization
        with open(stats_file, 'r') as f:
            self.stats = json.load(f)
            
        # Extract normalization arrays
        self._setup_normalization()
        
        if verbose:
            print(f"✅ Data loaded - {self.total_samples} samples available")
            print(f"   Input variables: {len(input_variables)}")
            print(f"   Output variables: {len(output_variables)}")
            print(f"   Ground truth available: {self.has_ground_truth()}")
    
    def _analyze_data_structure(self):
        """Analyze the data file structure to understand dimensions and groups."""
        try:
            # Check input group
            with xr.open_dataset(self.data_file, group='input') as ds:
                self.input_dataset_info = {
                    'dims': dict(ds.sizes),  # Use sizes instead of dims
                    'vars': list(ds.data_vars),
                    'sample_shape': None
                }
                if ds.data_vars:
                    sample_var = list(ds.data_vars)[0]
                    self.input_dataset_info['sample_shape'] = ds[sample_var].shape
                    # Extract coordinate information from actual dimensions
                    self.total_samples = ds.sizes['sample']
                    # Get spatial dimensions from the actual data
                    if 'y_lr' in ds.sizes and 'x_lr' in ds.sizes:
                        self.spatial_shape = (ds.sizes['y_lr'], ds.sizes['x_lr'])
                    else:
                        self.spatial_shape = ds[sample_var].shape[1:]  # fallback
                        
            # Check output group  
            try:
                with xr.open_dataset(self.data_file, group='output') as ds:
                    self.output_dataset_info = {
                        'dims': dict(ds.sizes),
                        'vars': list(ds.data_vars),
                        'sample_shape': None
                    }
                    if ds.data_vars:
                        sample_var = list(ds.data_vars)[0]
                        self.output_dataset_info['sample_shape'] = ds[sample_var].shape
                    self._has_output_group = True
                    
                    # Get output spatial dimensions
                    if 'y_hr' in ds.sizes and 'x_hr' in ds.sizes:
                        self.output_spatial_shape = (ds.sizes['y_hr'], ds.sizes['x_hr'])
                    else:
                        self.output_spatial_shape = self.spatial_shape  # fallback
                        
            except:
                self.output_dataset_info = None
                self._has_output_group = False
                self.output_spatial_shape = self.spatial_shape
                
            # Get time data from main dataset and create coordinate grids
            try:
                with xr.open_dataset(self.data_file) as ds:
                    # Load time data (it's a variable, not coordinate)
                    if 'time' in ds.data_vars:
                        self.time_values = ds['time'].values
                    else:
                        # Create dummy time values based on sample count
                        base_time = np.datetime64('2024-05-01T00:00:00')
                        self.time_values = base_time + np.arange(self.total_samples) * np.timedelta64(6, 'h')
                        
                    # FIXED: Create coordinate grids matching the working version
                    # These should match your actual domain coordinates
                    self.lat_values = np.linspace(19.0, 28.0, self.spatial_shape[0])
                    self.lon_values = np.linspace(116.0, 126.0, self.spatial_shape[1])
                    
                    # Output coordinates (may be different resolution)
                    self.output_lat_values = np.linspace(19.0, 28.0, self.output_spatial_shape[0])
                    self.output_lon_values = np.linspace(116.0, 126.0, self.output_spatial_shape[1])
                        
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Could not load coordinate info: {e}")
                # Fallback values
                base_time = np.datetime64('2024-05-01T00:00:00')
                self.time_values = base_time + np.arange(self.total_samples) * np.timedelta64(6, 'h')
                self.lat_values = np.linspace(19.0, 28.0, self.spatial_shape[0])
                self.lon_values = np.linspace(116.0, 126.0, self.spatial_shape[1])
                self.output_lat_values = np.linspace(19.0, 28.0, self.output_spatial_shape[0])
                self.output_lon_values = np.linspace(116.0, 126.0, self.output_spatial_shape[1])
                        
        except Exception as e:
            if self.verbose:
                print(f"❌ Error analyzing data structure: {e}")
            raise
            
        if self.verbose:
            print(f"   📊 Input group: {self.input_dataset_info}")
            if self._has_output_group:
                print(f"   📋 Output group: {self.output_dataset_info}")
            print(f"   📐 Input spatial shape: {self.spatial_shape}")
            print(f"   📐 Output spatial shape: {self.output_spatial_shape}")
            print(f"   ⏰ Time samples: {len(self.time_values)}")
            print(f"   🗺️  Input Lat range: [{self.lat_values[0]:.2f}, {self.lat_values[-1]:.2f}]")
            print(f"   🗺️  Input Lon range: [{self.lon_values[0]:.2f}, {self.lon_values[-1]:.2f}]")
    
    def _setup_normalization(self):
        """Setup normalization statistics."""
        try:
            # Input normalization
            input_mean = []
            input_std = []
            for var in self.input_variables:
                if 'input' in self.stats and var in self.stats['input']:
                    input_mean.append(self.stats['input'][var]['mean'])
                    input_std.append(self.stats['input'][var]['std'])
                elif var in self.stats:
                    input_mean.append(self.stats[var]['mean'])
                    input_std.append(self.stats[var]['std'])
                else:
                    input_mean.append(0.0)
                    input_std.append(1.0)
            
            self.input_mean = np.array(input_mean, dtype=np.float32).reshape(-1, 1, 1)
            self.input_std = np.array(input_std, dtype=np.float32).reshape(-1, 1, 1)
            
            # Output normalization (if available)
            output_mean = []
            output_std = []
            for var in self.output_variables:
                if 'output' in self.stats and var in self.stats['output']:
                    output_mean.append(self.stats['output'][var]['mean'])
                    output_std.append(self.stats['output'][var]['std'])
                elif var in self.stats:
                    output_mean.append(self.stats[var]['mean'])
                    output_std.append(self.stats[var]['std'])
                else:
                    output_mean.append(0.0)
                    output_std.append(1.0)
            
            self.output_mean = np.array(output_mean, dtype=np.float32).reshape(-1, 1, 1)
            self.output_std = np.array(output_std, dtype=np.float32).reshape(-1, 1, 1)
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Using default normalization: {e}")
            # Default normalization
            self.input_mean = np.zeros((len(self.input_variables), 1, 1), dtype=np.float32)
            self.input_std = np.ones((len(self.input_variables), 1, 1), dtype=np.float32)
            self.output_mean = np.zeros((len(self.output_variables), 1, 1), dtype=np.float32)
            self.output_std = np.ones((len(self.output_variables), 1, 1), dtype=np.float32)
    
    def __call__(self, time_array: np.ndarray, variables: List[str]) -> xr.DataArray:
        """Load data for given times and variables - FIXED to match working version."""
        if self.verbose:
            print(f"🔄 Loading data for {len(time_array)} times and {len(variables)} variables")
        
        # Map time to sample indices - for now use first samples
        # You should implement proper time-to-sample mapping based on your time data
        sample_indices = list(range(len(time_array)))
        
        # Check if these are input or output variables
        is_input_request = all(var in self.input_variables for var in variables)
        is_output_request = all(var in self.output_variables for var in variables)
        
        if is_input_request:
            # Load from input group
            with xr.open_dataset(self.data_file, group='input') as ds:
                # Load all requested variables for the sample indices
                var_arrays = []
                for var in variables:
                    if var in ds.data_vars:
                        # Get data for specific samples and transpose to correct shape
                        var_data = ds[var].isel(sample=sample_indices)  # Shape: (time, y_lr, x_lr)
                        var_arrays.append(var_data.values)
                    else:
                        raise ValueError(f"Input variable {var} not found in dataset")
                
                # Stack along variable dimension: [variable, time, lat, lon]
                stacked_data = np.stack(var_arrays, axis=0)
                
                # Create coordinate arrays - FIXED to be consistent
                coords = {
                    "variable": np.array(variables),
                    "time": time_array,
                    "lat": self.lat_values,
                    "lon": self.lon_values
                }
                
                # FIXED: Create DataArray with consistent dimension order
                result = xr.DataArray(
                    stacked_data,
                    dims=["variable", "time", "lat", "lon"],  # FIXED: variable first!
                    coords=coords,
                    attrs={"source": "SimpleDataSource", "group": "input"}
                )
                
        elif is_output_request and self._has_output_group:
            # Load from output group
            with xr.open_dataset(self.data_file, group='output') as ds:
                var_arrays = []
                for var in variables:
                    if var in ds.data_vars:
                        var_data = ds[var].isel(sample=sample_indices)  # Shape: (time, y_hr, x_hr)
                        var_arrays.append(var_data.values)
                    else:
                        raise ValueError(f"Output variable {var} not found in output group")
                
                # Stack along variable dimension: [variable, time, lat, lon]
                stacked_data = np.stack(var_arrays, axis=0)
                
                # Create coordinate arrays for output
                coords = {
                    "variable": np.array(variables),
                    "time": time_array,
                    "lat": self.output_lat_values,
                    "lon": self.output_lon_values
                }
                
                # FIXED: Create DataArray with consistent dimension order
                result = xr.DataArray(
                    stacked_data,
                    dims=["variable", "time", "lat", "lon"],  # FIXED: variable first!
                    coords=coords,
                    attrs={"source": "SimpleDataSource", "group": "output"}
                )
        else:
            raise ValueError(f"Variables {variables} not found or mixed input/output request not supported")
        
        if self.verbose:
            print(f"✅ Loaded DataArray with shape: {result.shape}")
            print(f"   Dimensions: {dict(result.sizes)}")
            
        return result
    
    def has_ground_truth(self) -> bool:
        """Check if dataset contains ground truth for output variables."""
        return self._has_output_group and self.output_dataset_info is not None