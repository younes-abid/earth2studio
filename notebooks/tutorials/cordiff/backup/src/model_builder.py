"""
Model building and management for CorrDiff experiments.
"""
import os
import time
import torch
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim


class EnsembleFogIndexCorrDiff(torch.nn.Module, DiagnosticModel):
    """Enhanced Ensemble CorrDiff model with auto-detection capabilities."""
    
    def __init__(self, regression_model, residual_model=None, config=None):
        super().__init__()
        self.regression_model = regression_model
        self.residual_model = residual_model
        self.config = config or {}
        
        # Extract configuration
        self.sampling_mode = self.config.get('sampling_mode', 'stochastic')
        self.num_steps = self.config.get('number_of_steps', 20)
        self.solver = self.config.get('solver', 'euler')
        self.hr_mean_conditioning = self.config.get('hr_mean_conditioning', True)
        self.verbose = self.config.get('verbose', True)
        
        # Grid coordinates - will be auto-detected if None
        self.input_grid = self.config.get('input_grid')
        self.output_grid = self.config.get('output_grid')
        
        # Variables
        self.input_variables = self.config.get('input_variables', [])
        self.output_variables = self.config.get('output_variables', [])
        
        # Ensemble parameters
        self.number_of_samples = self.config.get('num_ensembles', 8)
        self.ensemble_seeds = None
        
        # Initialize grids and normalization
        self._initialize_grids()
        
    def _initialize_grids(self):
        """Initialize input and output grids."""
        if self.input_grid is None:
            # Default grid - will be updated when data source is available
            self.input_lat = np.linspace(19.25, 28.0, 36)
            self.input_lon = np.linspace(116.0, 126.0, 40)
            if self.verbose:
                print("⚠️  Using default input grid")
        else:
            self.input_lat = np.linspace(*self.input_grid['lat'])
            self.input_lon = np.linspace(*self.input_grid['lon'])
            
        if self.output_grid is None:
            # Default grid
            self.output_lat = np.linspace(19.0, 28.0, 432)
            self.output_lon = np.linspace(116.0, 126.0, 432)
            if self.verbose:
                print("⚠️  Using default output grid")
        else:
            self.output_lat = np.linspace(*self.output_grid['lat'])
            self.output_lon = np.linspace(*self.output_grid['lon'])
    
    def update_normalization(self, data_source):
        """Update normalization parameters from data source."""
        if hasattr(data_source, 'input_mean'):
            self.register_buffer('in_center', torch.from_numpy(data_source.input_mean))
            self.register_buffer('in_scale', torch.from_numpy(data_source.input_std))
        else:
            # Default normalization
            self.register_buffer('in_center', torch.zeros(len(self.input_variables), 1, 1))
            self.register_buffer('in_scale', torch.ones(len(self.input_variables), 1, 1))
            
        if hasattr(data_source, 'output_mean'):
            self.register_buffer('out_center', torch.from_numpy(data_source.output_mean))
            self.register_buffer('out_scale', torch.from_numpy(data_source.output_std))
        else:
            # Default normalization
            self.register_buffer('out_center', torch.zeros(len(self.output_variables), 1, 1))
            self.register_buffer('out_scale', torch.ones(len(self.output_variables), 1, 1))
    
    def update_grids_from_data_source(self, data_source):
        """Update grid coordinates from data source if available."""
        if hasattr(data_source, 'input_coords'):
            coords = data_source.input_coords()
            if 'lat' in coords and 'lon' in coords:
                self.input_lat = coords['lat']
                self.input_lon = coords['lon']
                if self.verbose:
                    print(f"✅ Updated input grid from data source: {len(self.input_lat)}x{len(self.input_lon)}")
        
        if hasattr(data_source, 'output_coords'):
            coords = data_source.output_coords()
            if 'lat' in coords and 'lon' in coords:
                self.output_lat = coords['lat']
                self.output_lon = coords['lon']
                if self.verbose:
                    print(f"✅ Updated output grid from data source: {len(self.output_lat)}x{len(self.output_lon)}")
    
    def set_ensemble_seeds(self, base_seed=42):
        """Generate ensemble seeds for reproducible sampling."""
        start_time = time.time() if self.verbose else None
        
        np.random.seed(base_seed)
        self.ensemble_seeds = np.random.randint(0, 2**31, size=self.number_of_samples)
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f'🎲 Generated {len(self.ensemble_seeds)} ensemble seeds in {elapsed:.3f}s')
            print(f'   Seeds: {self.ensemble_seeds[:5]}...')
    
    def input_coords(self):
        """Get input coordinate specification."""
        return OrderedDict({
            'batch': np.empty(0),
            'variable': np.array(self.input_variables),
            'lat': self.input_lat,
            'lon': self.input_lon,
        })
    
    @batch_coords()
    def output_coords(self, input_coords):
        """Get output coordinate specification."""
        output_coords = OrderedDict({
            'batch': np.empty(0),
            'sample': np.arange(self.number_of_samples),
            'variable': np.array(self.output_variables),
            'lat': self.output_lat,
            'lon': self.output_lon,
        })
        
        # Validate input coordinates
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
        """Interpolate input to output resolution."""
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
        """Forward pass with ensemble generation."""
        start_time = time.time() if self.verbose else None
        
        # Interpolate and normalize
        x = self._interpolate(x)
        x = (x - self.in_center) / self.in_scale
        
        # Add sample dimension
        x = x.unsqueeze(0).repeat(self.number_of_samples, 1, 1, 1)
        
        # Create output tensors
        img_h, img_w = len(self.output_lat), len(self.output_lon)
        latents = torch.zeros(self.number_of_samples, len(self.output_variables), 
                             img_h, img_w, device=x.device)
        
        # Generate ensemble seeds if not set
        if self.ensemble_seeds is None:
            self.set_ensemble_seeds(self.config.get('seed_base', 42))
        
        if self.residual_model is not None:
            # Full diffusion mode
            if self.verbose:
                print(f'🔄 Running {self.sampling_mode} sampling with {self.num_steps} steps')
            
            # Import sampling functions
            from physicsnemo.utils.generative import StackedRandomGenerator, deterministic_sampler, stochastic_sampler
            
            # Create random number generator with ensemble seeds
            rnd = StackedRandomGenerator(x.device, torch.from_numpy(self.ensemble_seeds))
            noise = rnd.randn_like(latents)
            
            # Regression step (if using HR mean conditioning)
            if self.hr_mean_conditioning:
                mean_start = time.time() if self.verbose else None
                mean = self.regression_model(latents, x)
                if self.verbose:
                    print(f'   Regression completed in {time.time() - mean_start:.3f}s')
                # Concatenate input and regression mean for diffusion model
                x_with_mean = torch.cat([x, mean], dim=1)
            else:
                mean = torch.zeros_like(latents)
                x_with_mean = x
            
            # Choose sampling method
            diffusion_start = time.time() if self.verbose else None
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
            
            if self.verbose:
                print(f'   Diffusion completed in {time.time() - diffusion_start:.3f}s')
            
            x = mean + residual
        else:
            # Regression only
            if self.verbose:
                print('📊 Running regression-only mode')
            x = self.regression_model(latents, x)
        
        # Denormalize
        x = self.out_scale * x + self.out_center
        
        if self.verbose:
            total_time = time.time() - start_time
            print(f'✅ Forward pass completed in {total_time:.3f}s')
        
        return x
    
    @batch_func()
    def __call__(self, x, coords):
        """Main forward call with batching support."""
        output_coords = self.output_coords(coords)
        out = torch.zeros([len(v) for v in output_coords.values()], 
                         device=x.device, dtype=torch.float32)
        
        for i in range(out.shape[0]):
            out[i] = self._forward(x[i])
        
        return out, output_coords


def load_models(config, verbose=True):
    """Load CorrDiff models from checkpoints."""
    start_time = time.time() if verbose else None
    
    # Get checkpoint paths
    regression_path, diffusion_path = config.get_checkpoint_paths()
    
    if verbose:
        print(f'🔄 Loading models...')
        print(f'   Regression: {Path(regression_path).name}')
        print(f'   Diffusion: {Path(diffusion_path).name}')
    
    # Load regression model
    from physicsnemo.models import Module as PhysicsNemoModule
    
    model_start = time.time() if verbose else None
    regression_model = PhysicsNemoModule.from_checkpoint(regression_path).eval()
    if verbose:
        print(f'   ✅ Regression model loaded in {time.time() - model_start:.3f}s')
    
    # Load diffusion model (if exists)
    diffusion_model = None
    if os.path.exists(diffusion_path):
        model_start = time.time() if verbose else None
        diffusion_model = PhysicsNemoModule.from_checkpoint(diffusion_path).eval()
        if verbose:
            print(f'   ✅ Diffusion model loaded in {time.time() - model_start:.3f}s')
    else:
        if verbose:
            print('   ⚠️  No diffusion model found, using regression-only mode')
    
    # Create ensemble model
    ensemble_model = EnsembleFogIndexCorrDiff(
        regression_model, 
        diffusion_model, 
        config=config.config
    )
    
    # Set ensemble seeds for reproducibility
    ensemble_model.set_ensemble_seeds(config.get('seed_base', 42))
    
    if verbose:
        total_time = time.time() - start_time
        print(f'✅ All models loaded successfully in {total_time:.3f}s')
        print(f'🎯 Model mode: {"Full CorrDiff" if diffusion_model else "Regression-only"}')
        print(f'🎲 Ensemble members: {ensemble_model.number_of_samples}')
    
    return ensemble_model


def initialize_model_with_data(ensemble_model, data_source, verbose=True):
    """Initialize model with data source information."""
    if verbose:
        print('🔄 Initializing model with data source...')
    
    # Update grids from data source
    ensemble_model.update_grids_from_data_source(data_source)
    
    # Update normalization parameters
    ensemble_model.update_normalization(data_source)
    
    if verbose:
        print('✅ Model initialization complete')
    
    return ensemble_model