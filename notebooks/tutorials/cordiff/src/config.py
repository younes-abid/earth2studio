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
        self.DATA_FILE = self.BASE_DATA_PATH + 'ERA5_WRF_combined_concatenated_432/' + '2022-06-05_2022-11-22_150.nc' 
        # 2019-01-01_2019-06-09_150.nc  2020-01-06_2020-07-09_150.nc  2020-12-31_2021-06-27_150.nc  2021-12-23_2022-06-04_150.nc  2022-11-23_2023-05-13_150.nc  2023-10-26_2024-04-29_150.nc
        # 2019-06-10_2020-01-05_150.nc  2020-07-10_2020-12-30_150.nc  2021-06-28_2021-12-22_150.nc  2022-06-05_2022-11-22_150.nc  2023-05-14_2023-10-25_150.nc  2024-04-30_2024-05-30_21.nc   
        
        
        self.STATS_FILE = self.BASE_DATA_PATH + 'stats_432/stat.json'

        # Variables
        self.INPUT_VARIABLES = ['t_850', 't_500', 'z_850', 'z_500', 'u_850', 'u_500', 'v_850', 'v_500', 'u10', 'v10', 't2m', 'd2m', 'skt', 'sp', 'tcwv', 'tp']
        self.OUTPUT_VARIABLES = ['Rain_rate_LN','Rain_rate_PT']

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
        # July 27-29, 2022; Feb 11-13, 2024; March 08-10, 2024;  Apr 15-17, 2024; May 01-02, 2024
        self.INFERENCE_TIMES = [
            '2022-07-27T00:00:00', '2022-07-27T01:00:00', '2022-07-27T02:00:00', '2022-07-27T03:00:00',
            # '2022-07-27T04:00:00', '2022-07-27T05:00:00', '2022-07-27T06:00:00', '2022-07-27T07:00:00',
            # '2022-07-27T08:00:00', '2022-07-27T09:00:00', '2022-07-27T10:00:00', '2022-07-27T11:00:00',
            # '2022-07-27T12:00:00', '2022-07-27T13:00:00', '2022-07-27T14:00:00', '2022-07-27T15:00:00',
            # '2022-07-27T16:00:00', '2022-07-27T17:00:00', '2022-07-27T18:00:00', '2022-07-27T19:00:00',
            # '2022-07-27T20:00:00', '2022-07-27T21:00:00', '2022-07-27T22:00:00', '2022-07-27T23:00:00',
            
            # '2022-07-28T00:00:00', '2022-07-28T01:00:00', '2022-07-28T02:00:00', '2022-07-28T03:00:00',
            # '2022-07-28T04:00:00', '2022-07-28T05:00:00', '2022-07-28T06:00:00', '2022-07-28T07:00:00',
            # '2022-07-28T08:00:00', '2022-07-28T09:00:00', '2022-07-28T10:00:00', '2022-07-28T11:00:00',
            # '2022-07-28T12:00:00', '2022-07-28T13:00:00', '2022-07-28T14:00:00', '2022-07-28T15:00:00',
            # '2022-07-28T16:00:00', '2022-07-28T17:00:00', '2022-07-28T18:00:00', '2022-07-28T19:00:00',
            # '2022-07-28T20:00:00', '2022-07-28T21:00:00', '2022-07-28T22:00:00', '2022-07-28T23:00:00',
            
            # '2022-07-29T00:00:00', '2022-07-29T01:00:00', '2022-07-29T02:00:00', '2022-07-29T03:00:00',
            # '2022-07-29T04:00:00', '2022-07-29T05:00:00', '2022-07-29T06:00:00', '2022-07-29T07:00:00',
            # '2022-07-29T08:00:00', '2022-07-29T09:00:00', '2022-07-29T10:00:00', '2022-07-29T11:00:00',
            # '2022-07-29T12:00:00', '2022-07-29T13:00:00', '2022-07-29T14:00:00', '2022-07-29T15:00:00',
            # '2022-07-29T16:00:00', '2022-07-29T17:00:00', '2022-07-29T18:00:00', '2022-07-29T19:00:00',
            # '2022-07-29T20:00:00', '2022-07-29T21:00:00', '2022-07-29T22:00:00', '2022-07-29T23:00:00'
        ]
        
        # ['2024-04-30T00:00:00', '2024-05-15T06:00:00']

        # Ensemble parameters
        self.NUM_ENSEMBLES = 4 # Number of ensemble members to generate
        self.SEED_BASE = 42     # Base seed for reproducible ensemble generation

        # Sampling configuration
        self.SAMPLING_MODE = 'stochastic'  # 'deterministic' or 'stochastic'
        self.NUMBER_OF_STEPS = 18         # Number of diffusion sampling steps
        self.SOLVER = 'euler'              # Diffusion solver: 'euler' or 'heun'

        # High-resolution mean conditioning (recommended for better results)
        self.HR_MEAN_CONDITIONING = True

        # Plotting configuration
        self.VARIABLE_CMAP = "Blues"
        self.STD_CMAP = "magma"
        self.RESIDUAL_CMAP = "RdBu_r"  # Diverging colormap for residuals
        self.METRIC_CMAP = "viridis"   # For metric plots
        self.VARIABLE_VMIN = 0
        self.VARIABLE_VMAX = 100
        self.USE_VMIN_VMAX = False  # Whether to use vmin/vmax in plots
        self.USE_CARTOPY = True  # Whether to use Cartopy for geographic plots
        self.FIGURE_SIZE = (12, 8)  # Default figure size
        self.DPI = 150  # Figure DPI for high quality plots
        self.SAVE_FORMAT = 'png'  # Save format: png, pdf, svg

        # Plot types configuration
        self.PLOT_INPUT_VARIABLES = True  # Grid plot of all input variables used in pipeline
        self.PLOT_ENSEMBLE_MEMBERS = True  # Grid plot of all ensemble members  
        self.PLOT_ENSEMBLE_STATISTICS = True  # Ensemble statistics (mean, std, min, max)
        self.PLOT_ENSEMBLE_RESIDUALS = True  # Ensemble residuals vs ground truth
        self.PLOT_ENSEMBLE_RESIDUALS_STATISTICS = True  # Statistics of ensemble residuals
        self.PLOT_GROUND_TRUTH_VS_ENSEMBLEMEAN = True  # Ground truth vs ensemble mean comparison
        self.PLOT_UNCERTAINTY_QUANTIFICATION = False  # Prediction intervals and reliability (slow to compute)

        # Metrics configuration - Core regression metrics
        self.PLOT_CORE_METRICS = {
            'RMSE': {},     # Root Mean Square Error
            'MAE': {},      # Mean Absolute Error
            'R2': {},       # Coefficient of Determination
            'BIAS': {},     # Mean Bias
            'MAPE': {"epsilon": 1e-8}      # Mean Absolute Percentage Error (avoid division by zero)
        }

        # Image quality metrics (super-resolution)
        self.PLOT_IMAGE_QUALITY_METRICS = {
            'PSNR': {"data_range": 1.0},     # Peak Signal-to-Noise Ratio
            'SSIM': {"win_size": 11, "data_range": 1.0, "multichannel": False},     # Structural Similarity Index
            'MS_SSIM': {"win_size": 11, "data_range": 1.0, "weights": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]},     # Multi-Scale SSIM
            'LPIPS': {"net": "alex", "spatial": False}      # Learned Perceptual Image Patch Similarity
        }

        # Ensemble/Probabilistic metrics  
        self.PLOT_ENSEMBLE_METRICS = {
            'CRPS': {"ensemble_axis": 0},        # Continuous Ranked Probability Score
            'RELIABILITY': {"bins": 10, "alpha": 0.05}, # Reliability (calibration)
            'SHARPNESS': {"ensemble_axis": 0},   # Ensemble spread/sharpness
            'COVERAGE': {"confidence_levels": [0.5, 0.68, 0.9, 0.95]},    # Prediction interval coverage
            'PIT': {"bins": 10},         # Probability Integral Transform
            'RANK_HISTOGRAM': {"bins": "ensemble_size"}  # Rank histogram uniformity
        }

        # Diffusion-specific metrics
        self.PLOT_DIFFUSION_METRICS = {
            'FID': {"batch_size": 32, "dims": 2048, "device": "cuda"},         # Fréchet Inception Distance
            'IS': {"batch_size": 32, "splits": 10, "resize": True},          # Inception Score
            'KID': {"batch_size": 32, "subset_size": 1000},         # Kernel Inception Distance
            'PRECISION': {"k": 3, "batch_size": 32},   # Precision of generated samples
            'RECALL': {"k": 3, "batch_size": 32}       # Recall of generated samples
        }

        # Spectral/Frequency metrics
        self.PLOT_SPECTRAL_METRICS = {
            'PSD_ERROR': {"nperseg": 64, "noverlap": 32, "scaling": "density"},      # Power Spectral Density Error
            'SPECTRAL_ANGLE': {"normalize": True}, # Spectral Angle Mapper
            'GRADIENT_ERROR': {"axis": None, "edge_order": 1}, # Gradient magnitude error
            'FOURIER_LOSS': {"norm": "ortho", "axes": (-2, -1)}    # Fourier domain loss
        }

        # Spatial pattern metrics
        self.PLOT_PATTERN_METRICS = {
            'SAL': {"threshold_percentile": 95},           # Structure-Amplitude-Location
            'FSS': {"scales": [1, 3, 5, 9, 15], "threshold": 0.1},           # Fractions Skill Score
            'SPATIAL_CORR': {"method": "pearson"},  # Spatial correlation
            'OBJECT_BASED': {"threshold": 0.5, "min_area": 9}   # Object-based verification
        }

        
        # Output configuration
        self.TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.OUTPUT_ENSEMBLE_FILE = f'/app/outputs/generation/{self.VARIABLES}/{self.TIMESTAMP}/ensemble_{self.VARIABLES}_{self.TIMESTAMP}.nc'
        self.OUTPUT_ANALYSIS_FOLDER = f'/app/outputs/generation/{self.VARIABLES}/{self.TIMESTAMP}/analysis'
        self.SAVE_INPUT_VARIABLES = True  # Whether to save input variables in the output file
        self.SAVE_PREDICTIONS = True  # Whether to save predictions in the output file
        self.SAVE_GROUND_TRUTH = True  # Whether to save ground truth variables in the output file

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