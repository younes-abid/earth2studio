# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Ensemble inference workflow for CorrDiff models.
Contains functions to run ensemble inference and collect results.
"""

import torch
import numpy as np
from earth2studio.data import prep_data_array
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array


def run_ensemble_inference(config, corrdiff_model, data_source, device=None):
    """
    Run ensemble inference and collect results.
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object with inference settings
    corrdiff_model : EnsembleFogIndexCorrDiff
        Ensemble model for inference
    data_source : CustomCorrDiffDataSource
        Data source for loading input data
    device : torch.device, optional
        Device for computation (auto-detected if None)
    
    Returns
    -------
    dict
        Dictionary with predictions, inputs, coordinates, and times
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    times = config.INFERENCE_TIMES
    
    print(f'🚀 Starting ensemble inference on {device}')
    print(f'📅 Processing {len(times)} time steps')
    print(f'🎯 Generating {corrdiff_model.number_of_samples} ensemble members per time step')
    
    # Move model to device
    corrdiff_model = corrdiff_model.to(device)
    
    # Storage for all results
    all_predictions = []
    all_inputs = []
    all_coords = []
    processed_times = []
    
    # Process each time step
    for i, time_step in enumerate(times):
        print(f'🔄 Processing time {i+1}/{len(times)}: {time_step}')
        
        try:
            # Load data for this time step
            time_array = to_time_array([time_step])
            x, coords = prep_data_array(
                data_source(time_array, config.INPUT_VARIABLES), 
                device=device
            )
            x, coords = map_coords(x, coords, corrdiff_model.input_coords())
            
            print(f'   📊 Input data shape: {x.shape}')
            
            # Store input for later saving
            all_inputs.append(x.cpu())
            
            # Run ensemble inference
            with torch.no_grad():
                pred, pred_coords = corrdiff_model(x, coords)
            
            print(f'   ✅ Generated ensemble shape: {pred.shape}')
            
            # Store results
            all_predictions.append(pred.cpu())
            all_coords.append(pred_coords)
            processed_times.append(time_array[0])
            
        except Exception as e:
            print(f'   ❌ Error processing {time_step}: {e}')
            continue
    
    print(f'✅ Ensemble inference complete! Processed {len(processed_times)} time steps')
    
    return {
        'predictions': all_predictions,
        'inputs': all_inputs,
        'coords': all_coords,
        'times': processed_times
    }