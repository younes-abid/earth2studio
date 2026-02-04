"""
Simple CorrDiff ensemble model wrapper.
"""
import torch
import numpy as np
from typing import Dict, Any, Optional
from collections import OrderedDict

from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import handshake_coords, handshake_dim
from physicsnemo.utils.generative import StackedRandomGenerator, deterministic_sampler, stochastic_sampler


class SimpleCorrDiffEnsemble(torch.nn.Module, DiagnosticModel):
    """Simple CorrDiff ensemble model - FIXED to match working 1.1 version."""
    
    def __init__(self, regression_model, diffusion_model, data_source, config: Dict[str, Any]):
        super().__init__()
        self.regression_model = regression_model
        self.diffusion_model = diffusion_model
        self.data_source = data_source
        self.config = config
        self.verbose = config.get('verbose', False)
        
        # Extract config
        self.input_variables = config['input_variables']
        self.output_variables = config['output_variables']
        self.num_ensembles = config['num_ensembles']
        self.sampling_mode = config['sampling_mode']
        self.num_steps = config['number_of_steps']
        self.solver = config['solver']
        self.hr_mean_conditioning = config['hr_mean_conditioning']
        
        # FIXED: Use hardcoded grids like working version, fall back to data source grids
        input_grid = config.get('input_grid')
        output_grid = config.get('output_grid')
        
        if input_grid is not None:
            self.input_lat = np.linspace(*input_grid['lat'])
            self.input_lon = np.linspace(*input_grid['lon'])
        else:
            # Use data source grids directly
            self.input_lat = data_source.lat_values
            self.input_lon = data_source.lon_values
            
        if output_grid is not None:
            self.output_lat = np.linspace(*output_grid['lat'])
            self.output_lon = np.linspace(*output_grid['lon'])
        else:
            # Use data source grids directly (same as input for simplicity)
            self.output_lat = data_source.output_lat_values
            self.output_lon = data_source.output_lon_values
        
        # Setup normalization - FIXED to match working version
        self.register_buffer('in_center', torch.from_numpy(data_source.input_mean))
        self.register_buffer('in_scale', torch.from_numpy(data_source.input_std))
        self.register_buffer('out_center', torch.from_numpy(data_source.output_mean) if hasattr(data_source, 'output_mean') else torch.zeros(len(self.output_variables), 1, 1))
        self.register_buffer('out_scale', torch.from_numpy(data_source.output_std) if hasattr(data_source, 'output_std') else torch.ones(len(self.output_variables), 1, 1))
        
        # Store ensemble parameters for compatibility with working version
        self.number_of_samples = self.num_ensembles
        self.ensemble_seeds = None
        
        if self.verbose:
            print(f'✅ Model initialized - {self.num_ensembles} ensemble members')
            print(f'   Input grid: lat({len(self.input_lat)}), lon({len(self.input_lon)})')
            print(f'   Output grid: lat({len(self.output_lat)}), lon({len(self.output_lon)})')
    
    def set_ensemble_seeds(self, base_seed: int = 42):
        """Generate ensemble seeds."""
        np.random.seed(base_seed)
        self.ensemble_seeds = np.random.randint(0, 2**31, size=self.num_ensembles)
        if self.verbose:
            print(f'🎲 Generated {len(self.ensemble_seeds)} ensemble seeds')
    
    # FIXED: Use exact same structure as working 1.1 version
    def input_coords(self):
        return OrderedDict({
            'batch': np.empty(0),
            'variable': np.array(self.input_variables),
            'lat': self.input_lat,
            'lon': self.input_lon,
        })
    
    # FIXED: Use exact same structure as working 1.1 version
    @batch_coords()
    def output_coords(self, input_coords):
        output_coords = OrderedDict({
            'batch': np.empty(0),
            'sample': np.arange(self.num_ensembles),
            'variable': np.array(self.output_variables),
            'lat': self.output_lat,
            'lon': self.output_lon,
        })
        
        # Validate input - FIXED to match working version exactly
        target = self.input_coords()
        handshake_dim(input_coords, 'lon', 3)
        handshake_dim(input_coords, 'lat', 2)
        handshake_dim(input_coords, 'variable', 1)
        handshake_coords(input_coords, target, 'lon')
        handshake_coords(input_coords, target, 'lat')
        handshake_coords(input_coords, target, 'variable')
        
        output_coords['batch'] = input_coords['batch']
        return output_coords
    
    def _interpolate_to_output(self, x):
        """Interpolate input to output resolution if needed."""
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
        """Forward pass - FIXED to match working version."""
        if self.verbose:
            print(f'   🔄 Processing input shape: {x.shape}')
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Interpolate and normalize
        x = self._interpolate_to_output(x)
        x = (x - self.in_center) / self.in_scale
        
        # Add ensemble dimension
        x = x.unsqueeze(0).repeat(self.num_ensembles, 1, 1, 1)
        
        # Create latents
        img_h, img_w = len(self.output_lat), len(self.output_lon)
        latents = torch.zeros(
            self.num_ensembles, 
            len(self.output_variables),
            img_h, 
            img_w, 
            device=x.device
        )
        
        # Generate ensemble seeds if needed
        if self.ensemble_seeds is None:
            self.set_ensemble_seeds(self.config.get('seed_base', 42))
        
        if self.diffusion_model is not None:
            # Full CorrDiff mode
            if self.verbose:
                print(f'   🎯 Running {self.sampling_mode} sampling with {self.num_steps} steps')
            
            # Random generator
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
            
            # Diffusion sampling
            if self.sampling_mode == 'deterministic':
                residual = deterministic_sampler(
                    self.diffusion_model, 
                    noise, 
                    x_with_mean, 
                    randn_like=rnd.randn_like, 
                    num_steps=self.num_steps,
                    solver=self.solver
                )
            elif self.sampling_mode == 'stochastic':
                residual = stochastic_sampler(
                    self.diffusion_model, 
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
            if self.verbose:
                print('   📊 Running regression-only mode')
            x = self.regression_model(latents, x)
        
        # Clear intermediate tensors
        del latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Denormalize
        x = self.out_scale * x + self.out_center
        return x
    
    # FIXED: Use exact same @batch_func() approach as working 1.1 version
    @batch_func()
    def __call__(self, x, coords):
        """Call method using batch_func like working version."""
        if self.verbose:
            print(f'   📋 Model received tensor shape: {x.shape}')
            print(f'   📋 Model received coords: {list(coords.keys())}')
        
        # Use the exact same approach as working 1.1 version
        output_coords = self.output_coords(coords)
        out = torch.zeros([len(v) for v in output_coords.values()], device=x.device, dtype=torch.float32)
        
        # Process batches (should be 1 batch in our case)
        for i in range(out.shape[0]):
            # FIXED: Handle the input tensor properly like in working version
            result = self._forward(x[i])
            out[i] = result
            
            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.verbose:
            print(f'   ✅ Output tensor shape: {out.shape}')
        
        return out, output_coords
