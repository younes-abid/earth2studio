# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Ensemble CorrDiff Model for Earth2Studio.
Contains the main ensemble model class with configurable sampling.
"""

import os
import torch
import numpy as np
from collections import OrderedDict
from pathlib import Path

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim

from physicsnemo.models import Module as PhysicsNemoModule
from physicsnemo.utils.generative import StackedRandomGenerator, deterministic_sampler, stochastic_sampler


class EnsembleFogIndexCorrDiff(torch.nn.Module, DiagnosticModel):
    """Ensemble CorrDiff model for Fog_index predictions with configurable sampling"""
    
    def __init__(self, config, regression_model, residual_model=None, data_source=None):
        super().__init__()
        self.config = config
        self.regression_model = regression_model
        self.residual_model = residual_model
        
        # Sampling configuration
        sampling_config = config.get_sampling_config()
        self.sampling_mode = sampling_config['mode']
        self.num_steps = sampling_config['num_steps']
        self.solver = sampling_config['solver']
        self.hr_mean_conditioning = sampling_config['hr_mean_conditioning']
        
        # Grid coordinates
        self.input_lat = np.linspace(*config.INPUT_GRID['lat'])
        self.input_lon = np.linspace(*config.INPUT_GRID['lon'])
        self.output_lat = np.linspace(*config.OUTPUT_GRID['lat'])
        self.output_lon = np.linspace(*config.OUTPUT_GRID['lon'])
        
        # Normalization (extract from data source if provided)
        if data_source:
            self.register_buffer('in_center', torch.from_numpy(data_source.input_mean))
            self.register_buffer('in_scale', torch.from_numpy(data_source.input_std))
            self.register_buffer('out_center', torch.from_numpy(data_source.output_mean) if hasattr(data_source, 'output_mean') else torch.zeros(len(config.OUTPUT_VARIABLES), 1, 1))
            self.register_buffer('out_scale', torch.from_numpy(data_source.output_std) if hasattr(data_source, 'output_std') else torch.ones(len(config.OUTPUT_VARIABLES), 1, 1))
        else:
            # Default normalization
            self.register_buffer('in_center', torch.zeros(len(config.INPUT_VARIABLES), 1, 1))
            self.register_buffer('in_scale', torch.ones(len(config.INPUT_VARIABLES), 1, 1))
            self.register_buffer('out_center', torch.zeros(len(config.OUTPUT_VARIABLES), 1, 1))
            self.register_buffer('out_scale', torch.ones(len(config.OUTPUT_VARIABLES), 1, 1))
        
        # Ensemble parameters
        self.number_of_samples = config.NUM_ENSEMBLES
        self.ensemble_seeds = None  # Will be set during inference
    
    def set_ensemble_seeds(self, base_seed=None):
        """Generate ensemble seeds for reproducible sampling"""
        base_seed = base_seed or self.config.SEED_BASE
        np.random.seed(base_seed)
        self.ensemble_seeds = np.random.randint(0, 2**31, size=self.number_of_samples)
        print(f'🎲 Generated {len(self.ensemble_seeds)} ensemble seeds: {self.ensemble_seeds[:5]}...')
    
    def input_coords(self):
        return OrderedDict({
            'batch': np.empty(0),
            'variable': np.array(self.config.INPUT_VARIABLES),
            'lat': self.input_lat,
            'lon': self.input_lon,
        })
    
    @batch_coords()
    def output_coords(self, input_coords):
        output_coords = OrderedDict({
            'batch': np.empty(0),
            'sample': np.arange(self.number_of_samples),
            'variable': np.array(self.config.OUTPUT_VARIABLES),
            'lat': self.output_lat,
            'lon': self.output_lon,
        })
        
        # Validate input
        target = self.input_coords()
        handshake_dim(input_coords, 'lon', 3)
        handshake_dim(input_coords, 'lat', 2)
        handshake_dim(input_coords, 'variable', 1)
        handshake_coords(input_coords, target, 'lon')
        handshake_coords(input_coords, target, 'lat')
        handshake_coords(input_coords, target, 'variable')
        
        output_coords['batch'] = input_coords['batch']
        return output_coords
    
    def _interpolate(self, x):
        """Interpolate to output resolution"""
        if x.shape[-2:] == (len(self.output_lat), len(self.output_lon)):
            return x
        
        # Add batch dim if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            added_batch = True
        else:
            added_batch = False
        
        # Interpolate
        import torch.nn.functional as F
        target_size = (len(self.output_lat), len(self.output_lon))
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        
        # Remove batch dim if added
        if added_batch:
            x = x.squeeze(0)
        
        return x
    
    @torch.inference_mode()
    def _forward(self, x):
        """Forward pass with ensemble generation"""
        # Interpolate and normalize
        x = self._interpolate(x)
        x = (x - self.in_center) / self.in_scale
        
        # Add sample dimension
        x = x.unsqueeze(0).repeat(self.number_of_samples, 1, 1, 1)
        
        # Create output tensors
        img_h, img_w = len(self.output_lat), len(self.output_lon)
        latents = torch.zeros(self.number_of_samples, len(self.config.OUTPUT_VARIABLES), img_h, img_w, device=x.device)
        
        # Generate ensemble seeds if not set
        if self.ensemble_seeds is None:
            self.set_ensemble_seeds()
        
        if self.residual_model is not None:
            # Full diffusion mode
            print(f'🔄 Running {self.sampling_mode} sampling with {self.num_steps} steps')
            
            # Create random number generator with ensemble seeds
            rnd = StackedRandomGenerator(x.device, torch.from_numpy(self.ensemble_seeds))
            noise = rnd.randn_like(latents)
            
            # Regression step (if using HR mean conditioning)
            if self.hr_mean_conditioning:
                mean = self.regression_model(latents, x)
                # Concatenate input and regression mean for diffusion model
                x_with_mean = torch.cat([x, mean], dim=1)
            else:
                mean = torch.zeros_like(latents)
                x_with_mean = x
            
            # Choose sampling method
            if self.sampling_mode == 'deterministic':
                residual = deterministic_sampler(
                    self.residual_model, 
                    noise, 
                    x_with_mean, 
                    randn_like=rnd.randn_like, 
                    num_steps=self.num_steps,
                    solver=self.solver
                )
            elif self.sampling_mode == 'stochastic':
                residual = stochastic_sampler(
                    self.residual_model, 
                    noise, 
                    x_with_mean, 
                    randn_like=rnd.randn_like, 
                    num_steps=self.num_steps
                )
            else:
                raise ValueError(f'Unknown sampling mode: {self.sampling_mode}')
            
            x = mean + residual
        else:
            # Regression only
            print('📊 Running regression-only mode')
            x = self.regression_model(latents, x)
        
        # Denormalize
        x = self.out_scale * x + self.out_center
        return x
    
    @batch_func()
    def __call__(self, x, coords):
        output_coords = self.output_coords(coords)
        out = torch.zeros([len(v) for v in output_coords.values()], device=x.device, dtype=torch.float32)
        
        for i in range(out.shape[0]):
            out[i] = self._forward(x[i])
        
        return out, output_coords


def load_models(config):
    """
    Load regression and diffusion models from checkpoints.
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object with model paths
    
    Returns
    -------
    tuple
        (regression_model, diffusion_model)
    """
    # Load regression model
    print(f'🔄 Loading regression model: {Path(config.REGRESSION_CHECKPOINT).name}')
    regression_model = PhysicsNemoModule.from_checkpoint(config.REGRESSION_CHECKPOINT).eval()
    
    # Load diffusion model (if exists)
    diffusion_model = None
    if os.path.exists(config.DIFFUSION_CHECKPOINT):
        print(f'🔄 Loading diffusion model: {Path(config.DIFFUSION_CHECKPOINT).name}')
        diffusion_model = PhysicsNemoModule.from_checkpoint(config.DIFFUSION_CHECKPOINT).eval()
    else:
        print('⚠️  No diffusion model found, using regression-only mode')
    
    return regression_model, diffusion_model


def create_ensemble_model(config, data_source):
    """
    Create ensemble CorrDiff model from configuration.
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object
    data_source : CustomCorrDiffDataSource
        Data source for normalization statistics
    
    Returns
    -------
    EnsembleFogIndexCorrDiff
        Ensemble model ready for inference
    """
    # Load models
    regression_model, diffusion_model = load_models(config)
    
    # Create ensemble CorrDiff model
    ensemble_corrdiff = EnsembleFogIndexCorrDiff(
        config=config,
        regression_model=regression_model, 
        residual_model=diffusion_model, 
        data_source=data_source
    )
    
    # Set ensemble seeds for reproducibility
    ensemble_corrdiff.set_ensemble_seeds(config.SEED_BASE)
    
    print('✅ Ensemble models loaded successfully')
    print(f'🎯 Model mode: {"Full CorrDiff" if diffusion_model else "Regression-only"}')
    print(f'🎲 Sampling: {config.SAMPLING_MODE} with {config.NUM_ENSEMBLES} ensemble members')
    
    return ensemble_corrdiff