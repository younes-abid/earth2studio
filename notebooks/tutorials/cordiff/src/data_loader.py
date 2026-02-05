# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Optimized Data Loader for CorrDiff workflows.
Focuses on data loading without coordinate bureaucracy.
"""

import json
import time as time_pkg
from typing import List, Optional
from pathlib import Path

import numpy as np
import xarray as xr
from earth2studio.data.base import DataSource
from earth2studio.utils.time import to_time_array


class CustomCorrDiffDataSource(DataSource):
    """Streamlined DataSource for CorrDiff datasets."""
    
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
        
        self._setup_data_info()
        self._load_stats()
        
        if cache_data:
            self._preload_data()
    
    def _setup_data_info(self):
        """Setup basic file and grid information"""
        self.file_sizes = []
        self.cumulative_sizes = []
        total_size = 0
        
        for file_path in self.data_paths:
            with xr.open_dataset(file_path, group="input") as ds:
                size = ds.sizes["sample"]
                self.file_sizes.append(size)
                total_size += size
                self.cumulative_sizes.append(total_size)
                
        self.total_samples = total_size
        
        # Get grid shape from first file
        with xr.open_dataset(self.data_paths[0], group="input") as ds:
            sample_data = ds[self.input_variables[0]]
            self.input_shape = sample_data.shape[-2:]  # (lat, lon)
            
        # Simple coordinate arrays for Earth2Studio interface
        self.lat_input = np.linspace(19.25, 28, self.input_shape[0], endpoint=True)
        self.lon_input = np.linspace(116, 126, self.input_shape[1], endpoint=False)
        
        # Load time coordinates
        with xr.open_dataset(self.data_paths[0]) as ds:
            if "time" in ds.variables:
                self.time_values = np.array(ds["time"])
            else:
                self.time_values = np.arange(self.file_sizes[0])
    
    def _load_stats(self):
        """Load normalization statistics"""
        try:
            with open(self.stats_path, "r") as f:
                stats = json.load(f)
            
            self.input_mean, self.input_std = self._extract_stats(stats, self.input_variables, "input")
            
            if self.output_variables:
                self.output_mean, self.output_std = self._extract_stats(stats, self.output_variables, "output")
                
        except Exception as e:
            # Default stats if loading fails
            self.input_mean = np.zeros((len(self.input_variables), 1, 1), dtype=np.float32)
            self.input_std = np.ones((len(self.input_variables), 1, 1), dtype=np.float32)
            
            if self.output_variables:
                self.output_mean = np.zeros((len(self.output_variables), 1, 1), dtype=np.float32)
                self.output_std = np.ones((len(self.output_variables), 1, 1), dtype=np.float32)
    
    def _extract_stats(self, stats, variables, group):
        """Extract stats for variables"""
        mean = np.array([stats[group][v]["mean"] for v in variables])[:, None, None].astype(np.float32)
        std = np.array([stats[group][v]["std"] for v in variables])[:, None, None].astype(np.float32)
        return mean, std
    
    def _preload_data(self):
        """Preload data into memory"""
        self.cached_inputs = []
        self.cached_times = []
        
        for file_path in self.data_paths:
            # Load input data
            input_data, _ = self._load_dataset_group(file_path, "input", self.input_variables)
            self.cached_inputs.append(input_data)
            
            # Load time data
            with xr.open_dataset(file_path) as ds:
                if "time" in ds.variables:
                    time_data = np.array(ds["time"])
                else:
                    time_data = np.arange(len(input_data))
                self.cached_times.append(time_data)
    
    def _load_dataset_group(self, file_path, group, variables, stack_axis=1):
        """Load data from NetCDF group"""
        with xr.open_dataset(file_path, group=group) as ds:
            data = np.stack([ds[v] for v in variables], axis=stack_axis)
        return data, variables
    
    def __call__(self, time, variable):
        """Load data for specified times and variables (Earth2Studio interface)"""
        time_array = to_time_array(time)
        time_indices = self._find_time_indices(time_array)
        
        # Load data for requested variables
        all_var_data = []
        for var in variable:
            if var in self.input_variables:
                var_data = self._get_variable_data(var, time_indices)
                all_var_data.append(var_data)
            else:
                # Create dummy data for missing variables
                dummy_shape = (len(time_array), len(self.lat_input), len(self.lon_input))
                all_var_data.append(np.zeros(dummy_shape, dtype=np.float32))
        
        # Stack variables
        stacked_data = np.stack(all_var_data, axis=1)  # [time, variable, lat, lon]
        
        # Create DataArray with minimal coordinates
        coords = {
            "time": time_array,
            "variable": np.array(variable),
            "lat": self.lat_input,
            "lon": self.lon_input
        }
        
        return xr.DataArray(
            stacked_data,
            dims=["time", "variable", "lat", "lon"],
            coords=coords,
            attrs={"source": "CustomCorrDiffDataSource"}
        )
    
    def _find_time_indices(self, requested_times):
        """Find indices for requested times"""
        if self.time_values is None:
            return list(range(len(requested_times)))
        
        indices = []
        for req_time in requested_times:
            req_time_np = np.datetime64(req_time)
            time_diffs = np.abs(self.time_values - req_time_np)
            closest_idx = np.argmin(time_diffs)
            indices.append(closest_idx)
        
        return indices
    
    def _get_variable_data(self, variable, time_indices):
        """Get data for specific variable at time indices"""
        if not self.cache_data:
            raise NotImplementedError("On-demand loading not implemented")
        
        var_idx = self.input_variables.index(variable)
        all_data = []
        
        for time_idx in time_indices:
            # Find file containing this time index
            file_idx = next(i for i, size in enumerate(self.cumulative_sizes) if time_idx < size)
            
            # Get relative index within file
            relative_idx = time_idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)
            
            # Get data from cache
            sample_data = self.cached_inputs[file_idx][relative_idx, var_idx]
            all_data.append(sample_data)
        
        return np.array(all_data)


def create_data_source(config):
    """Create data source from configuration"""
    return CustomCorrDiffDataSource(
        data_paths=[config.DATA_FILE],
        stats_path=config.STATS_FILE,
        input_variables=config.INPUT_VARIABLES,
        output_variables=config.OUTPUT_VARIABLES,
        cache_data=True
    )