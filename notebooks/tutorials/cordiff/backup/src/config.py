"""
Configuration management for CorrDiff experiments.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any


class CorrDiffConfig:
    """Configuration manager for CorrDiff experiments."""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize configuration."""
        self.config = config_dict or {}
        self._setup_defaults()
    
    def _setup_defaults(self):
        """Set default configuration values."""
        defaults = {
            # Model configuration
            'variables': 'Fog_index',
            'input_variables': [
                't_850', 't_500', 'z_850', 'z_500', 'u_850', 'u_500', 
                'v_850', 'v_500', 'u10', 'v10', 't2m', 'd2m', 'skt', 
                'sp', 'tcwv', 'tp'
            ],
            'output_variables': ['Fog_index'],
            
            # Grid configuration
            'input_grid': None,  # Will be auto-detected if None
            'output_grid': None,  # Will be auto-detected if None
            
            # Ensemble configuration
            'num_ensembles': 8,
            'seed_base': 42,
            'sampling_mode': 'stochastic',
            'number_of_steps': 20,
            'solver': 'euler',
            'hr_mean_conditioning': True,
            
            # Paths (to be updated by user)
            'base_checkpoints_path': '/app/host/home/younes.abid/git/physicsnemo/outputs/checkpoints/',
            'base_data_path': '/app/host/mnt/storage/younes.abid/physicsnemo/data/custom_data_2/',
            'output_base_path': '/app/outputs/generation/',
            
            # Processing configuration
            'verbose': True,
            'inference_times': ['2024-05-01T00:00:00'],
            
            # Visualization configuration
            'enable_plots': True,
            'plot_types': ['ensemble_members', 'ensemble_stats', 'metrics', 'histograms'],
        }
        
        # Update with provided config
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def update(self, config_dict: Dict):
        """Update configuration with new values."""
        self.config.update(config_dict)
    
    def auto_detect_grids(self, data_source):
        """Auto-detect input and output grids from data source."""
        if hasattr(data_source, 'input_shape') and self.config['input_grid'] is None:
            # Try to auto-detect grid from data source
            print("⚠️  Input grid not specified, attempting auto-detection...")
            
        if self.config['output_grid'] is None:
            print("⚠️  Output grid not specified, attempting auto-detection...")
    
    def get_checkpoint_paths(self):
        """Get checkpoint paths based on configuration."""
        variables = self.config['variables']
        base_path = self.config['base_checkpoints_path']
        
        regression_path = os.path.join(
            base_path, variables, 'checkpoints_regression', self.config['checkpoint_regression']
        )
        diffusion_path = os.path.join(
            base_path, variables, 'checkpoints_diffusion', self.config['checkpoint_diffusion']
        )
        
        return regression_path, diffusion_path
    
    def get_data_paths(self):
        """Get data file paths based on configuration."""
        base_path = self.config['base_data_path']
        
        data_file = os.path.join(
            base_path, self.config['data_file']
        )
        stats_file = os.path.join(
            base_path, self.config['stat_file']
        )
        
        return data_file, stats_file
    
    def get_output_paths(self):
        """Get output paths with timestamp."""
        variables = self.config['variables']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_dir = os.path.join(
            self.config['output_base_path'], variables, timestamp
        )
        
        output_file = os.path.join(output_dir, 'ensemble.nc')
        plot_dir = os.path.join(output_dir, 'plots')
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        return output_file, plot_dir
    
    def validate_config(self):
        """Validate configuration and check file existence."""
        issues = []
        
        # Check checkpoint paths
        regression_path, diffusion_path = self.get_checkpoint_paths()
        if not os.path.exists(regression_path):
            issues.append(f"Regression checkpoint not found: {regression_path}")
        
        if not os.path.exists(diffusion_path):
            print(f"⚠️  Diffusion checkpoint not found: {diffusion_path}")
            print("    Will use regression-only mode")
        
        # Check data paths
        data_file, stats_file = self.get_data_paths()
        if not os.path.exists(data_file):
            issues.append(f"Data file not found: {data_file}")
        
        if not os.path.exists(stats_file):
            issues.append(f"Stats file not found: {stats_file}")
        
        if issues:
            raise FileNotFoundError("\n".join(issues))
        
        return True
    
    def print_summary(self):
        """Print configuration summary."""
        print("🔧 Configuration Summary")
        print("=" * 60)
        print(f"Variables: {self.config['variables']}")
        print(f"Ensemble members: {self.config['num_ensembles']}")
        print(f"Sampling mode: {self.config['sampling_mode']}")
        print(f"Diffusion steps: {self.config['number_of_steps']}")
        print(f"HR mean conditioning: {self.config['hr_mean_conditioning']}")
        print(f"Verbose mode: {self.config['verbose']}")
        print(f"Enable plots: {self.config['enable_plots']}")
        
        # Show paths
        regression_path, diffusion_path = self.get_checkpoint_paths()
        data_file, stats_file = self.get_data_paths()
        output_file, plot_dir = self.get_output_paths()
        
        print(f"\n📁 Paths:")
        print(f"Regression: {Path(regression_path).name}")
        print(f"Diffusion: {Path(diffusion_path).name}")
        print(f"Data: {Path(data_file).name}")
        print(f"Output: {output_file}")
        print("=" * 60)
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def load_config(self, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)
            self.config.update(loaded_config)


def create_default_config() -> CorrDiffConfig:
    """Create a default configuration instance."""
    return CorrDiffConfig()


def load_config_from_file(filepath: str) -> CorrDiffConfig:
    """Load configuration from file."""
    config = CorrDiffConfig()
    config.load_config(filepath)
    return config