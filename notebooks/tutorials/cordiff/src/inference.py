# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Optimized ensemble inference workflow for CorrDiff models.
Streamlined for efficient batch processing.
"""

import torch
import numpy as np
from earth2studio.data import prep_data_array
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array


def run_ensemble_inference(config, corrdiff_model, data_source, device=None):
    """
    Run ensemble inference and collect results.
    
    Returns dict with predictions, inputs, coords, and times.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    times = config.INFERENCE_TIMES
    corrdiff_model = corrdiff_model.to(device)
    
    # Storage for results
    all_predictions = []
    all_inputs = []
    all_coords = []
    processed_times = []
    
    print(f'🚀 Running inference: {len(times)} time steps, {corrdiff_model.number_of_samples} ensemble members on {device} device')
    
    # Process each time step
    for i, time_step in enumerate(times):
        try:
            # Load and prepare data
            time_array = to_time_array([time_step])
            x, coords = prep_data_array(
                data_source(time_array, config.INPUT_VARIABLES), 
                device=device
            )
            x, coords = map_coords(x, coords, corrdiff_model.input_coords())
            
            # Store input
            all_inputs.append(x.cpu())
            
            # Run inference
            with torch.no_grad():
                pred, pred_coords = corrdiff_model(x, coords)
            
            # Store results
            all_predictions.append(pred.cpu())
            all_coords.append(pred_coords)
            processed_times.append(time_array[0])
            
            if (i + 1) % max(1, len(times) // 10) == 0:  # Progress every 10%
                print(f'   Progress: {i+1}/{len(times)} steps completed')
            
        except Exception as e:
            print(f'❌ Error at step {i+1}: {e}')
            continue
    
    print(f'✅ Inference complete: {len(processed_times)}/{len(times)} steps successful')
    
    return {
        'predictions': all_predictions,
        'inputs': all_inputs,
        'coords': all_coords,
        'times': processed_times
    }