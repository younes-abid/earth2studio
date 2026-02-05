# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Optimized Ensemble CorrDiff Model for Earth2Studio.
Streamlined for any variable predictions with minimal coordinate bureaucracy.
"""

import os
import torch
import numpy as np
import psutil
import time
from collections import OrderedDict
from pathlib import Path

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim

from physicsnemo.models import Module as PhysicsNemoModule
from physicsnemo.utils.generative import StackedRandomGenerator, deterministic_sampler, stochastic_sampler
from trim_coordinates import load_trimmed_coordinates


def get_model_info(model, model_type, load_time):
    """Get model statistics and memory usage"""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get memory usage in MB
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    print(f"✅ {model_type} model loaded in {load_time:.2f}s")
    print(f"   📊 Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"   💾 Memory usage: {memory_mb:.1f} MB")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_mb': memory_mb,
        'load_time': load_time
    }


class EnsembleCorrDiff(torch.nn.Module, DiagnosticModel):
    """Streamlined Ensemble CorrDiff model for any variable predictions"""
    
    def __init__(self, config, regression_model, residual_model=None, data_source=None):
        super().__init__()
        self.config = config
        self.regression_model = regression_model
        self.residual_model = residual_model
        self.number_of_samples = config.NUM_ENSEMBLES
        
        # Sampling configuration
        sampling_config = config.get_sampling_config()
        self.sampling_mode = sampling_config['mode']
        self.num_steps = sampling_config['num_steps']
        self.solver = sampling_config['solver']
        self.hr_mean_conditioning = sampling_config['hr_mean_conditioning']
        
        # Simple coordinate arrays for Earth2Studio interface
        self.input_lat = np.linspace(*config.INPUT_GRID['lat'])
        self.input_lon = np.linspace(*config.INPUT_GRID['lon'])
        self.output_lat = np.linspace(*config.OUTPUT_GRID['lat'])
        self.output_lon = np.linspace(*config.OUTPUT_GRID['lon'])
        
        # Store 2D coordinates for output saving (if using WRF)
        if config.USE_WRF_COORDINATES:
            self.output_lat_2d, self.output_lon_2d = load_trimmed_coordinates(config.WRF_COORD_FILE)
        else:
            self.output_lat_2d = None
            self.output_lon_2d = None
        
        # Setup normalization
        self._setup_normalization(data_source, config)
        
        # Generate ensemble seeds
        np.random.seed(config.SEED_BASE)
        self.ensemble_seeds = np.random.randint(0, 2**31, size=self.number_of_samples)
    
    def _setup_normalization(self, data_source, config):
        """Setup normalization buffers"""
        if data_source:
            self.register_buffer('in_center', torch.from_numpy(data_source.input_mean))
            self.register_buffer('in_scale', torch.from_numpy(data_source.input_std))
            if hasattr(data_source, 'output_mean'):
                self.register_buffer('out_center', torch.from_numpy(data_source.output_mean))
                self.register_buffer('out_scale', torch.from_numpy(data_source.output_std))
            else:
                self.register_buffer('out_center', torch.zeros(len(config.OUTPUT_VARIABLES), 1, 1))
                self.register_buffer('out_scale', torch.ones(len(config.OUTPUT_VARIABLES), 1, 1))
        else:
            # Default normalization
            self.register_buffer('in_center', torch.zeros(len(config.INPUT_VARIABLES), 1, 1))
            self.register_buffer('in_scale', torch.ones(len(config.INPUT_VARIABLES), 1, 1))
            self.register_buffer('out_center', torch.zeros(len(config.OUTPUT_VARIABLES), 1, 1))
            self.register_buffer('out_scale', torch.ones(len(config.OUTPUT_VARIABLES), 1, 1))
    
    def input_coords(self):
        return OrderedDict({
            'batch': np.empty(0),
            'variable': np.array(self.config.INPUT_VARIABLES),
            'lat': self.input_lat,
            'lon': self.input_lon,
        })
    
    @batch_coords()
    def output_coords(self, input_coords):
        # Basic validation (required by Earth2Studio)
        target = self.input_coords()
        handshake_dim(input_coords, 'lon', 3)
        handshake_dim(input_coords, 'lat', 2)
        handshake_dim(input_coords, 'variable', 1)
        handshake_coords(input_coords, target, 'lon')
        handshake_coords(input_coords, target, 'lat')
        handshake_coords(input_coords, target, 'variable')
        
        output_coords = OrderedDict({
            'batch': input_coords['batch'],
            'sample': np.arange(self.number_of_samples),
            'variable': np.array(self.config.OUTPUT_VARIABLES),
            'lat': self.output_lat,
            'lon': self.output_lon,
        })
        
        return output_coords
    
    def _interpolate(self, x):
        """Interpolate to output resolution"""
        target_shape = (len(self.output_lat), len(self.output_lon))
        if x.shape[-2:] == target_shape:
            return x
        
        import torch.nn.functional as F
        needs_batch = len(x.shape) == 3
        if needs_batch:
            x = x.unsqueeze(0)
        
        x = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=True)
        
        if needs_batch:
            x = x.squeeze(0)
        
        return x
    
    @torch.inference_mode()
    def _forward(self, x):
        """Forward pass with ensemble generation"""
        # Prepare input
        x = self._interpolate(x)
        x = (x - self.in_center) / self.in_scale
        x = x.unsqueeze(0).repeat(self.number_of_samples, 1, 1, 1)
        
        # Create latents
        img_h, img_w = len(self.output_lat), len(self.output_lon)
        latents = torch.zeros(self.number_of_samples, len(self.config.OUTPUT_VARIABLES), 
                            img_h, img_w, device=x.device)
        
        if self.residual_model is not None:
            # Full diffusion mode
            rnd = StackedRandomGenerator(x.device, torch.from_numpy(self.ensemble_seeds))
            noise = rnd.randn_like(latents)
            
            if self.hr_mean_conditioning:
                mean = self.regression_model(latents, x)
                x_with_mean = torch.cat([x, mean], dim=1)
            else:
                mean = torch.zeros_like(latents)
                x_with_mean = x
            
            # Sample residual
            if self.sampling_mode == 'deterministic':
                residual = deterministic_sampler(
                    self.residual_model, noise, x_with_mean, 
                    randn_like=rnd.randn_like, num_steps=self.num_steps, solver=self.solver
                )
            else:  # stochastic
                residual = stochastic_sampler(
                    self.residual_model, noise, x_with_mean, 
                    randn_like=rnd.randn_like, num_steps=self.num_steps
                )
            
            x = mean + residual
        else:
            # Regression only
            x = self.regression_model(latents, x)
        
        # Denormalize and return
        return self.out_scale * x + self.out_center
    
    @batch_func()
    def __call__(self, x, coords):
        output_coords = self.output_coords(coords)
        out = torch.zeros([len(v) for v in output_coords.values()], device=x.device, dtype=torch.float32)
        
        for i in range(out.shape[0]):
            out[i] = self._forward(x[i])
        
        return out, output_coords


def load_models(config):
    """Load regression and diffusion models with proper security and statistics"""
    # Load regression model with timing
    start_time = time.time()
    regression_model = PhysicsNemoModule.from_checkpoint(config.REGRESSION_CHECKPOINT).eval()
    regression_load_time = time.time() - start_time
    
    # Get regression model info
    regression_info = get_model_info(regression_model, "Regression", regression_load_time)
    
    # Load diffusion model (if exists)
    diffusion_model = None
    diffusion_info = None
    if os.path.exists(config.DIFFUSION_CHECKPOINT):
        start_time = time.time()
        diffusion_model = PhysicsNemoModule.from_checkpoint(config.DIFFUSION_CHECKPOINT).eval()
        diffusion_load_time = time.time() - start_time
        
        # Get diffusion model info
        diffusion_info = get_model_info(diffusion_model, "Diffusion", diffusion_load_time)
    else:
        print("ℹ️  No diffusion model found - using regression-only mode")
    
    return regression_model, diffusion_model


def create_ensemble_model(config, data_source):
    """Create ensemble model from configuration"""
    regression_model, diffusion_model = load_models(config)
    
    return EnsembleCorrDiff(
        config=config,
        regression_model=regression_model, 
        residual_model=diffusion_model, 
        data_source=data_source
    )