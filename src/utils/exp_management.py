"""
Experiment Management
=====================

Functions for managing experiments and configurations.
"""

import os
from datetime import datetime
from pathlib import Path


def create_experiment_name(architecture, loss_name):
    """Create experiment name from configuration (without timestamp)"""
    return f"{architecture}_{loss_name}"


def setup_loss_configs():
    """Setup different loss configurations for experiments"""
    from losses import MSELoss, SSIMLoss, MultiScaleSSIMLoss as MS_SSIMLoss, SobelGradientLoss, FocalFrequencyLoss
    
    loss_configs = {
        'mse': {
            'mse': {
                'class': MSELoss,
                'weight': 1.0
            }
        },
        'ssim': {
            'ssim': {
                'class': SSIMLoss,
                'weight': 1.0,
                'params': {'window_size': 11, 'sigma': 1.5}
            }
        },
        'mse_ssim': {
            'mse': {
                'class': MSELoss,
                'weight': 0.2
            },
            'ssim': {
                'class': SSIMLoss,
                'weight': 0.8,
                'params': {'window_size': 11}
            }
        },
        'focal_freq': {
            'focal_freq': {
                'class': FocalFrequencyLoss,
                'weight': 1.0,
                'params': {
                    'alpha': 1.0,
                    'patch_factor': 1,
                    'ave_spectrum': False,
                    'log_matrix': False,
                    'batch_matrix': False
                }
            }
        },
        'mse_focal_freq': {
            'mse': {
                'class': MSELoss,
                'weight': 0.4
            },
            'focal_freq': {
                'class': FocalFrequencyLoss,
                'weight': 0.6,
                'params': {
                    'alpha': 1.0,
                    'patch_factor': 1,
                    'ave_spectrum': False,
                    'log_matrix': False,
                    'batch_matrix': False
                }
            }
        },
        'mse_ssim_focal_freq': {
            'mse': {
                'class': MSELoss,
                'weight': 0.2
            },
            'ssim': {
                'class': SSIMLoss,
                'weight': 0.4,
                'params': {'window_size': 11}
            },
            'focal_freq': {
                'class': FocalFrequencyLoss,
                'weight': 0.4,
                'params': {
                    'alpha': 1.0,
                    'patch_factor': 1,
                    'ave_spectrum': False,
                    'log_matrix': False,
                    'batch_matrix': False
                }
            }
        },
        'comprehensive': {
            'mse': {
                'class': MSELoss,
                'weight': 0.3
            },
            'ssim': {
                'class': SSIMLoss,
                'weight': 0.3,
                'params': {'window_size': 11, 'sigma': 1.5}
            },
            'focal_freq': {
                'class': FocalFrequencyLoss,
                'weight': 0.2,
                'params': {'alpha': 1.0, 'patch_factor': 1}
            },
            'sobel': {
                'class': SobelGradientLoss,
                'weight': 0.2
            }
        },
        'ms_ssim': {
            'ms_ssim': {
                'class': MS_SSIMLoss,
                'weight': 1.0,
                'params': {'window_size': 11}
            }
        },
        'mse_ms_ssim': {
            'mse': {
                'class': MSELoss,
                'weight': 0.4
            },
            'ms_ssim': {
                'class': MS_SSIMLoss,
                'weight': 0.6,
                'params': {'window_size': 11}
            }
        },
        'sobel': {
            'sobel': {
                'class': SobelGradientLoss,
                'weight': 1.0
            }
        },
        'mse_sobel': {
            'mse': {
                'class': MSELoss,
                'weight': 0.5
            },
            'sobel': {
                'class': SobelGradientLoss,
                'weight': 0.5
            }
        },
        'ssim_sobel': {
            'ssim': {
                'class': SSIMLoss,
                'weight': 0.6,
                'params': {'window_size': 11}
            },
            'sobel': {
                'class': SobelGradientLoss,
                'weight': 0.4
            }
        },
        'ms_ssim_sobel': {
            'ms_ssim': {
                'class': MS_SSIMLoss,
                'weight': 0.6,
                'params': {'window_size': 11}
            },
            'sobel': {
                'class': SobelGradientLoss,
                'weight': 0.4
            }
        },
        'mse_ms_ssim_sobel': {
            'mse': {
                'class': MSELoss,
                'weight': 0.3
            },
            'ms_ssim': {
                'class': MS_SSIMLoss,
                'weight': 0.4,
                'params': {'window_size': 11}
            },
            'sobel': {
                'class': SobelGradientLoss,
                'weight': 0.3
            }
        }
    }
    return loss_configs


def create_experiment_directories(base_output_dir, experiment_name):
    """Create all necessary directories for an experiment"""
    experiment_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    dirs = {
        'history': os.path.join(experiment_dir, 'history'),
        'weights': os.path.join(experiment_dir, 'weights'),
        'evaluation': os.path.join(experiment_dir, 'evaluation'),
        'experiment': experiment_dir
    }
    
    for dir_path in dirs.values():
        if dir_path != experiment_dir:  # experiment_dir already created
            os.makedirs(dir_path, exist_ok=True)
    
    # Create checkpoints subdirectory under weights
    dirs['checkpoints'] = os.path.join(dirs['weights'], 'checkpoints')
    os.makedirs(dirs['checkpoints'], exist_ok=True)
    
    return dirs


def create_session_summary(session_dir, session_timestamp, config, experiments):
    """Create a summary file for the entire experiment session"""
    session_summary_path = os.path.join(session_dir, 'session_summary.txt')
    with open(session_summary_path, 'w') as f:
        f.write(f"Experiment Session Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Session timestamp: {session_timestamp}\n")
        f.write(f"Device: {config['device']}\n")
        f.write(f"Image size: {config['image_size']}\n")
        f.write(f"Epochs: {config['num_epochs']}\n")
        f.write(f"Batch size: {config['batch_size']}\n\n")
        f.write(f"Experiments conducted:\n")
        for arch, loss in experiments:
            f.write(f"  - {arch}_{loss}\n")