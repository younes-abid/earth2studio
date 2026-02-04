# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-LicenseCopyrightText: Apache-2.0

"""
Custom Data Loader for Earth2Studio
====================================

This module adapts the PhysicsNeMo CustomDataset to work with Earth2Studio's DataSource interface.
Based on /home/younes.abid/git/physicsnemo/examples/weather/corrdiff/datasets/custom_list_2.py

Key Adaptations:
- Uses Earth2Studio DataSource base class
- Returns xarray.Dataset format expected by Earth2Studio
- Handles coordinate system mapping
- Maintains normalization capabilities from original
"""

import datetime
import time as time_pkg
import math
import json
from typing import List, Tuple, Union, Optional
from pathlib import Path

import numpy as np
from numba import jit, prange
import xarray as xr
import pandas as pd

from earth2studio.data.base import DataSource
from earth2studio.utils.time import to_time_array


class CustomCorrDiffDataSource(DataSource):
    """
    Earth2Studio DataSource for custom CorrDiff datasets.
    
    This class loads data from your PhysicsNeMo training format and converts it 
    to the Earth2Studio expected format.
    
    Parameters
    ----------
    data_paths : List[str]
        List of paths to NetCDF files with input/output groups
    stats_path : str  
        Path to JSON file containing normalization statistics
    input_variables : List[str]
        List of input variable names
    output_variables : List[str], optional
        List of output variable names (for reference)
    cache_data : bool, optional
        Whether to preload all data into memory (default: True)
    """
    
    def __init__(
        self,
        data_paths: List[str],
        stats_path: str,
        input_variables: List[str],
        output_variables: Optional[List[str]] = None,
        cache_data: bool = True
    ):
        self.data_paths = data_paths
        self.stats_path = stats_path  
        self.input_variables = input_variables
        self.output_variables = output_variables or []
        self.cache_data = cache_data
        
        # Initialize data structures
        self._setup_data_info()
        self._load_normalization_stats()
        
        if cache_data:
            self._preload_data()
    
    def _setup_data_info(self):
        """Set up file information and cumulative sizes"""
        self.file_sizes = []
        self.cumulative_sizes = []
        total_size = 0
        
        print("🔍 Analyzing data files...")
        for i, file_path in enumerate(self.data_paths):
            try:
                with xr.open_dataset(file_path, group="input") as ds:
                    size = ds.sizes["sample"]
                    self.file_sizes.append(size)
                    total_size += size
                    self.cumulative_sizes.append(total_size)
                    print(f"  File {i+1}: {Path(file_path).name} - {size} samples")
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
                raise
                
        self.total_samples = total_size
        print(f"✅ Total samples across all files: {total_size}")
        
        # Get coordinate information from first file
        self._load_coordinate_info()
    
    def _load_coordinate_info(self):
        """Load coordinate information from the first file"""
        with xr.open_dataset(self.data_paths[0]) as ds:
            # Load time coordinates
            if "time" in ds.variables:
                self.time_values = np.array(ds["time"])
            else:
                print("⚠️  No time coordinate found, using dummy values")
                self.time_values = np.arange(self.file_sizes[0])
                
            # Load spatial coordinates (if available)
            if "coord" in ds.variables:
                self.coord_data = np.array(ds["coord"])
            else:
                print("⚠️  No spatial coordinates found")
                self.coord_data = None
                
        # Get grid information from input data
        with xr.open_dataset(self.data_paths[0], group="input") as ds:
            sample_data = ds[self.input_variables[0]]
            self.input_shape = sample_data.shape[-2:]  # (lat, lon)
            
            # Create dummy lat/lon grids - you should replace with actual coordinates
            # Based on your PhysicsNeMo setup
            self.lat_input = np.linspace(19.25, 28, self.input_shape[0], endpoint=True)
            self.lon_input = np.linspace(116, 126, self.input_shape[1], endpoint=False)
            
        print(f"📐 Input grid shape: {self.input_shape}")
        print(f"📐 Lat range: [{self.lat_input[0]:.2f}, {self.lat_input[-1]:.2f}]")
        print(f"📐 Lon range: [{self.lon_input[0]:.2f}, {self.lon_input[-1]:.2f}]")
    
    def _load_normalization_stats(self):
        """Load normalization statistics from JSON file"""
        try:
            with open(self.stats_path, "r") as f:
                stats = json.load(f)
            
            self.input_mean, self.input_std = self._extract_stats(
                stats, self.input_variables, "input"
            )
            
            if self.output_variables:
                self.output_mean, self.output_std = self._extract_stats(
                    stats, self.output_variables, "output" 
                )
            
            print(f"✅ Loaded normalization stats for {len(self.input_variables)} input variables")
            
        except Exception as e:
            print(f"⚠️  Could not load normalization stats: {e}")
            print("   Using default normalization (mean=0, std=1)")
            self._create_default_stats()
    
    def _extract_stats(self, stats, variables, group):
        """Extract mean and std for given variables and group"""
        mean = np.array([stats[group][v]["mean"] for v in variables])[:, None, None].astype(np.float32)
        std = np.array([stats[group][v]["std"] for v in variables])[:, None, None].astype(np.float32)
        return mean, std
    
    def _create_default_stats(self):
        """Create default normalization statistics"""
        self.input_mean = np.zeros((len(self.input_variables), 1, 1), dtype=np.float32)
        self.input_std = np.ones((len(self.input_variables), 1, 1), dtype=np.float32)
        
        if self.output_variables:
            self.output_mean = np.zeros((len(self.output_variables), 1, 1), dtype=np.float32) 
            self.output_std = np.ones((len(self.output_variables), 1, 1), dtype=np.float32)
    
    def _preload_data(self):
        """Preload all data into memory for faster access"""
        print("💾 Preloading data into memory...")
        
        self.cached_inputs = []
        self.cached_outputs = []
        self.cached_times = []
        
        for file_idx, file_path in enumerate(self.data_paths):
            print(f"  Loading file {file_idx + 1}/{len(self.data_paths)}: {Path(file_path).name}")
            
            start_time = time_pkg.time()
            
            # Load input data
            input_data, _ = self._load_dataset_group(file_path, "input", self.input_variables)
            self.cached_inputs.append(input_data)
            
            # Load output data if available
            if self.output_variables:
                try:
                    output_data, _ = self._load_dataset_group(file_path, "output", self.output_variables)
                    self.cached_outputs.append(output_data)
                except Exception as e:
                    print(f"    ⚠️  Could not load output data: {e}")
                    self.cached_outputs.append(None)
            
            # Load time data
            with xr.open_dataset(file_path) as ds:
                if "time" in ds.variables:
                    time_data = np.array(ds["time"])
                else:
                    time_data = np.arange(self.file_sizes[file_idx])
                self.cached_times.append(time_data)
            
            elapsed = time_pkg.time() - start_time
            print(f"    ✅ Loaded in {elapsed:.2f}s")
        
        print("✅ Data preloading complete!")
    
    def _load_dataset_group(self, file_path, group, variables, stack_axis=1):
        """Load data from a specific group in the NetCDF file"""
        with xr.open_dataset(file_path, group=group) as ds:
            if variables is None:
                variables = list(ds.keys())
            data = np.stack([ds[v] for v in variables], axis=stack_axis)
        return data, variables
    
    def __call__(self, time, variable):
        """
        Load data for specified times and variables (Earth2Studio interface)
        
        Parameters
        ----------
        time : list[str] | list[datetime] | list[np.datetime64]
            List of times to load
        variable : list[str] 
            List of variables to load
            
        Returns
        -------
        xarray.DataArray
            DataArray with requested variables and times in Earth2Studio format
            (NOT Dataset - this was the bug!)
        """
        print(f"🔄 Loading data for {len(time)} times and {len(variable)} variables")
        
        # Convert time to standard format
        time_array = to_time_array(time)
        
        # Find matching time indices
        time_indices = self._find_time_indices(time_array)
        
        # Load data for requested variables
        all_var_data = []
        for var in variable:
            if var in self.input_variables:
                var_data = self._get_variable_data(var, time_indices)
                all_var_data.append(var_data)
            else:
                print(f"⚠️  Variable {var} not found in available variables: {self.input_variables}")
                # Create dummy data for missing variables
                dummy_shape = (len(time_array), len(self.lat_input), len(self.lon_input))
                all_var_data.append(np.zeros(dummy_shape, dtype=np.float32))
        
        # Stack all variables along a new 'variable' dimension
        stacked_data = np.stack(all_var_data, axis=1)  # Shape: [time, variable, lat, lon]
        
        # Create coordinate arrays
        coords = {
            "time": time_array,
            "variable": np.array(variable),
            "lat": self.lat_input,
            "lon": self.lon_input
        }
        
        # Create xarray DataArray (NOT Dataset!)
        da = xr.DataArray(
            stacked_data,
            dims=["time", "variable", "lat", "lon"],
            coords=coords,
            attrs={"source": "CustomCorrDiffDataSource"}
        )
        
        print(f"✅ Loaded DataArray with shape: {da.shape}")
        return da
    
    def _find_time_indices(self, requested_times):
        """Find indices for requested times"""
        # For now, return first few indices - you should implement proper time matching
        # based on your actual time coordinate system
        return list(range(min(len(requested_times), self.total_samples)))
    
    def _get_variable_data(self, variable, time_indices):
        """Get data for a specific variable at given time indices"""
        if not self.cache_data:
            return self._load_variable_on_demand(variable, time_indices)
        
        # Use cached data
        var_idx = self.input_variables.index(variable)
        all_data = []
        
        for time_idx in time_indices:
            # Find which file contains this time index
            file_idx = next(i for i, size in enumerate(self.cumulative_sizes) if time_idx < size)
            
            # Get relative index within the file
            relative_idx = time_idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)
            
            # Get data from cached arrays
            sample_data = self.cached_inputs[file_idx][relative_idx, var_idx]
            all_data.append(sample_data)
        
        return np.array(all_data)
    
    def _load_variable_on_demand(self, variable, time_indices):
        """Load variable data on-demand without caching"""
        # Implementation for loading data without preloading
        # You can implement this if memory is limited
        raise NotImplementedError("On-demand loading not implemented yet")
    
    def normalize_input(self, data, variable_names):
        """Normalize input data using loaded statistics"""
        normalized = np.zeros_like(data)
        
        for i, var in enumerate(variable_names):
            if var in self.input_variables:
                var_idx = self.input_variables.index(var)
                normalized[..., i, :, :] = (
                    (data[..., i, :, :] - self.input_mean[var_idx]) / self.input_std[var_idx]
                )
        
        return normalized
    
    def denormalize_output(self, data, variable_names):
        """Denormalize output data using loaded statistics"""
        if not hasattr(self, 'output_mean'):
            return data
            
        denormalized = np.zeros_like(data)
        
        for i, var in enumerate(variable_names):
            if var in self.output_variables:
                var_idx = self.output_variables.index(var)
                denormalized[..., i, :, :] = (
                    data[..., i, :, :] * self.output_std[var_idx] + self.output_mean[var_idx]
                )
        
        return denormalized


@jit(nopython=True)
def zoom_extrapolate(x, y, factor):
    """
    Bilinear zoom with extrapolation (from original PhysicsNeMo code)
    Use a numba function for performance.
    """
    s = 1 / factor
    for k in prange(y.shape[0]):
        for iy in range(y.shape[1]):
            ix = (iy + 0.5) * s - 0.5
            ix0 = int(math.floor(ix))
            ix0 = max(0, min(ix0, x.shape[1] - 2))
            ix1 = ix0 + 1
            for jy in range(y.shape[2]):
                jx = (jy + 0.5) * s - 0.5
                jx0 = int(math.floor(jx))
                jx0 = max(0, min(jx0, x.shape[2] - 2))
                jx1 = jx0 + 1

                x00 = x[k, ix0, jx0]
                x01 = x[k, ix0, jx1]
                x10 = x[k, ix1, jx0]
                x11 = x[k, ix1, jx1]
                djx = jx - jx0
                x0 = x00 + djx * (x01 - x00)
                x1 = x10 + djx * (x11 - x10)
                y[k, iy, jy] = x0 + (ix - ix0) * (x1 - x0)


# Helper function to create data source from your specific paths
def create_fog_index_data_source(
    base_path: str = "/app/host/home/younes.abid/git/physicsnemo/data",
    checkpoint_base: str = "/app/host/home/younes.abid/git/physicsnemo/outputs/checkpoints/Fog_index"
):
    """
    Convenience function to create a data source for your Fog_index experiment.
    Adjust paths and variables based on your actual setup.
    """
    
    # Example paths - adjust these to your actual file locations
    data_paths = [
        f"{base_path}/fog_data_file_1.nc",  # Replace with your actual files
        f"{base_path}/fog_data_file_2.nc", 
    ]
    
    stats_path = f"{checkpoint_base}/normalization_stats.json"
    
    # Example variables - adjust to match your Fog_index training
    input_variables = [
        'tcwv', 'z500', 't500', 'u500', 'v500', 
        'z850', 't850', 'u850', 'v850', 
        't2m', 'u10m', 'v10m'
    ]
    
    output_variables = [
        'fog_index',  # Your specific output variable
        't2m', 'u10m', 'v10m'
    ]
    
    return CustomCorrDiffDataSource(
        data_paths=data_paths,
        stats_path=stats_path,
        input_variables=input_variables,
        output_variables=output_variables,
        cache_data=True
    )