# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration module for CorrDiff ensemble inference.
Contains all configurable parameters for the ensemble workflow.
"""

from datetime import datetime
from pathlib import Path

class EnsembleConfig:
    """Configuration class for ensemble CorrDiff inference"""
    
    def __init__(self, variables="Fog_index"):
        # =============================================================================
        # ENSEMBLE CONFIGURATION - Update these paths and parameters for your setup
        # =============================================================================

        self.VARIABLES = variables

        # Checkpoint paths
        self.BASE_CHECKPOINTS_PATH = "/app/host/home/younes.abid/git/physicsnemo/outputs/checkpoints/"
        self.REGRESSION_CHECKPOINT = self.BASE_CHECKPOINTS_PATH + self.VARIABLES + '/checkpoints_regression/UNet.0.390000.mdlus'
        self.DIFFUSION_CHECKPOINT = self.BASE_CHECKPOINTS_PATH + self.VARIABLES + '/checkpoints_diffusion/EDMPrecondSuperResolution.0.590000.mdlus'

        # Data paths
        self.BASE_DATA_PATH = '/app/host/mnt/storage/younes.abid/physicsnemo/data/custom_data_2/'
        self.DATA_FILE = self.BASE_DATA_PATH + 'ERA5_WRF_combined_concatenated_432/2024-04-30_2024-05-30_21.nc'
        self.STATS_FILE = self.BASE_DATA_PATH + 'stats_432/stat.json'

        # Variables
        self.INPUT_VARIABLES = ['t_850', 't_500', 'z_850', 'z_500', 'u_850', 'u_500', 'v_850', 'v_500', 'u10', 'v10', 't2m', 'd2m', 'skt', 'sp', 'tcwv', 'tp']
        self.OUTPUT_VARIABLES = ['Fog_index']

        # Domain coordinates (adjust to your training domain)
        self.INPUT_GRID = {
            'lat': (19.25, 28.0, 36),   # (min, max, points)
            'lon': (116.0, 126.0, 40)   # (min, max, points)
        }
        self.OUTPUT_GRID = {
            'lat': (19.0, 28.0, 432),   # (min, max, points) 
            'lon': (116.0, 126.0, 432)  # (min, max, points)
        }

        # Coordinate system configuration
        self.USE_WRF_COORDINATES = True  # True: use WRF coordinates, False: use INPUT_GRID/OUTPUT_GRID
        self.WRF_COORD_FILE = '/app/host/home/younes.abid/git/physicsnemo/data/georefrenced/Space42_CorrDiff/trimmed_coordinates_432x432.nc'

        # =============================================================================
        # ENSEMBLE PARAMETERS - Key ensemble generation settings
        # =============================================================================

        # Inference settings
        self.INFERENCE_TIMES = ['2024-04-30T00:00:00', '2024-05-15T06:00:00']

        # Ensemble parameters
        self.NUM_ENSEMBLES = 4 # Number of ensemble members to generate
        self.SEED_BASE = 42     # Base seed for reproducible ensemble generation

        # Sampling configuration
        self.SAMPLING_MODE = 'stochastic'  # 'deterministic' or 'stochastic'
        self.NUMBER_OF_STEPS = 2          # Number of diffusion sampling steps
        self.SOLVER = 'euler'              # Diffusion solver: 'euler' or 'heun'

        # High-resolution mean conditioning (recommended for better results)
        self.HR_MEAN_CONDITIONING = True

        # Plotting configuration
        self.VARIABLE_CMAP = "Blues"
        self.STD_CMAP = "magma"
        self.VARIABLE_VMIN = 0
        self.VARIABLE_VMAX = 100

        # Output configuration
        self.TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.OUTPUT_ENSEMBLE_FILE = f'/app/outputs/generation/{self.VARIABLES}/{self.TIMESTAMP}/ensemble_{self.VARIABLES}_{self.TIMESTAMP}.nc'
        self.OUTPUT_ANALYSIS_FOLDER = f'/app/outputs/generation/{self.VARIABLES}/{self.TIMESTAMP}/analysis'

    def print_config(self):
        """Print configuration summary"""
        print('✅ Ensemble configuration loaded')
        print(f'📊 Ensemble setup: {self.NUM_ENSEMBLES} members, {self.SAMPLING_MODE} sampling')
        print(f'🔄 Diffusion steps: {self.NUMBER_OF_STEPS}, solver: {self.SOLVER}')
        print(f'💾 Ensemble output: {self.OUTPUT_ENSEMBLE_FILE}')
        print(f'📁 Analysis folder: {self.OUTPUT_ANALYSIS_FOLDER}')

    def get_sampling_config(self):
        """Get sampling configuration dictionary"""
        return {
            'mode': self.SAMPLING_MODE,
            'num_steps': self.NUMBER_OF_STEPS,
            'solver': self.SOLVER,
            'hr_mean_conditioning': self.HR_MEAN_CONDITIONING
        }