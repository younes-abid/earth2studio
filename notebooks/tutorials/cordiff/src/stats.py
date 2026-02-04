# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Fast ensemble statistics computation for CorrDiff results.
Minimal, efficient stats focusing on key summary numbers.
"""

import os
import json
import xarray as xr
from datetime import datetime
from config import EnsembleConfig

def compute_ensemble_summary_stats(config: EnsembleConfig, netcdf_path: str) -> dict:
    """
    Compute fast, minimal ensemble statistics - just essential summary numbers.
    
    For each time_idx and output_variable, compute min/max/mean/std of ensemble members.
    Returns only scalar summary values, not full spatial fields.
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object
    netcdf_path : str
        Path to saved NetCDF file
    
    Returns
    -------
    dict
        Dictionary with minimal summary statistics
    """
    print('📊 Computing fast ensemble summary statistics...')
    
    with xr.open_dataset(netcdf_path, group='prediction') as pred_ds:
        summary_stats = {
            'metadata': {
                'num_ensembles': config.NUM_ENSEMBLES,
                'num_times': len(pred_ds.time),
                'variables': config.OUTPUT_VARIABLES,
                'computed_at': datetime.now().isoformat()
            },
            'statistics': {}
        }
        
        for var_name in config.OUTPUT_VARIABLES:
            if var_name not in pred_ds:
                continue
                
            var_data = pred_ds[var_name]  # Shape: (ensemble, time, y, x)
            summary_stats['statistics'][var_name] = {}
            
            # For each time step, compute ensemble statistics
            for time_idx in range(len(pred_ds.time)):
                time_data = var_data.isel(time=time_idx)  # Shape: (ensemble, y, x)
                
                # Compute statistics across ensemble dimension only
                # Results are spatial fields that we then summarize to scalars
                ensemble_mean = time_data.mean(dim='ensemble')
                ensemble_std = time_data.std(dim='ensemble')
                ensemble_min = time_data.min(dim='ensemble') 
                ensemble_max = time_data.max(dim='ensemble')
                
                # Extract scalar summaries of the spatial fields
                time_stats = {
                    'ensemble_mean': {
                        'spatial_mean': float(ensemble_mean.mean()),
                        'spatial_std': float(ensemble_mean.std()),
                        'spatial_min': float(ensemble_mean.min()),
                        'spatial_max': float(ensemble_mean.max())
                    },
                    'ensemble_std': {
                        'spatial_mean': float(ensemble_std.mean()),
                        'spatial_max': float(ensemble_std.max()),
                        'spatial_min': float(ensemble_std.min())
                    },
                    'ensemble_range': {
                        'spatial_mean': float((ensemble_max - ensemble_min).mean()),
                        'spatial_max': float((ensemble_max - ensemble_min).max())
                    },
                    'raw_ensemble_values': {
                        'min_of_all_members': float(time_data.min()),
                        'max_of_all_members': float(time_data.max()),
                        'mean_of_all_members': float(time_data.mean())
                    }
                }
                
                summary_stats['statistics'][var_name][f'time_{time_idx}'] = time_stats
                
                print(f'   {var_name} time {time_idx}: '
                      f'mean={time_stats["ensemble_mean"]["spatial_mean"]:.4f}, '
                      f'uncertainty={time_stats["ensemble_std"]["spatial_mean"]:.4f}')
    
    return summary_stats


def save_summary_stats(config: EnsembleConfig, summary_stats: dict) -> str:
    """
    Save ensemble summary statistics to JSON file.
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object
    summary_stats : dict
        Summary statistics from compute_ensemble_summary_stats
    
    Returns
    -------
    str
        Path to saved statistics file
    """
    # Create analysis folder
    os.makedirs(config.OUTPUT_ANALYSIS_FOLDER, exist_ok=True)
    
    # Extract timestamp from ensemble file path
    stats_file = f"{config.OUTPUT_ANALYSIS_FOLDER}/summary_stats_{config.VARIABLES}_{config.TIMESTAMP}.json"
    
    # Save to JSON (already in JSON-serializable format)
    with open(stats_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f'✅ Summary statistics saved to {stats_file}')
    return stats_file


def compute_and_save_stats(config, netcdf_path):
    """
    Convenience function to compute and save ensemble statistics in one call.
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object
    netcdf_path : str
        Path to saved NetCDF file
        
    Returns
    -------
    tuple
        (summary_stats_dict, saved_file_path)
    """
    summary_stats = compute_ensemble_summary_stats(config, netcdf_path)
    stats_file = save_summary_stats(config, summary_stats)
    return summary_stats, stats_file