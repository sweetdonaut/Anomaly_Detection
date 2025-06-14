"""
Experiment Utilities
====================

Utilities for experiment management and configuration.
"""

import torch
import os
import json
from pathlib import Path
from datetime import datetime


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def create_experiment_name(architecture, loss_name):
    """Create experiment name from configuration (without timestamp)"""
    return f"{architecture}_{loss_name}"


def setup_loss_configs():
    """Setup different loss configurations for experiments"""
    from losses import MSELoss, SSIMLoss, MultiScaleSSIMLoss, SobelGradientLoss, FocalFrequencyLoss
    
    loss_configs = {
        'mse': {
            'mse': {
                'class': MSELoss,
                'weight': 1.0
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


def save_experiment_config(config, save_path):
    """Save experiment configuration to JSON file"""
    # Convert config to serializable format
    serializable_config = config.copy()
    serializable_config['device'] = str(config['device'])
    
    # Create a copy of loss config for serialization
    serializable_loss_config = {}
    for loss_name, loss_cfg in config['loss_config'].items():
        serializable_loss_config[loss_name] = {
            'class': loss_cfg['class'].__name__,
            'weight': loss_cfg['weight']
        }
        if 'params' in loss_cfg:
            serializable_loss_config[loss_name]['params'] = loss_cfg['params']
    
    serializable_config['loss_config'] = serializable_loss_config
    
    # Convert any Path objects to strings
    serializable_config['save_path'] = str(serializable_config.get('save_path', ''))
    
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def save_training_summary(train_history, config, experiment_name, save_path):
    """Save human-readable training summary"""
    with open(save_path, 'w') as f:
        f.write(f"Training Summary for {experiment_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Architecture: {config['architecture']}\n")
        f.write(f"Loss functions: {list(config['loss_config'].keys())}\n")
        f.write(f"Epochs: {config['num_epochs']}\n")
        f.write(f"Batch size: {config['batch_size']}\n")
        f.write(f"Learning rate: {config['lr']}\n")
        f.write(f"Image size: {config['image_size']}\n\n")
        
        f.write("Final Loss Values:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total loss: {train_history['total_loss'][-1]:.6f}\n")
        for loss_name, values in train_history['component_losses'].items():
            f.write(f"{loss_name}: {values[-1]:.6f}\n")
        
        f.write("\nLoss Weights:\n")
        f.write("-" * 30 + "\n")
        final_weights = train_history['weights'][-1]
        for loss_name, weight in final_weights.items():
            f.write(f"{loss_name}: {weight:.3f}\n")


def save_training_history_csv(train_history, save_path):
    """Save training history as CSV"""
    import pandas as pd
    
    # Prepare data for CSV
    history_data = {
        'epoch': list(range(1, len(train_history['total_loss']) + 1)),
        'total_loss': train_history['total_loss']
    }
    
    # Add component losses
    for loss_name, values in train_history['component_losses'].items():
        history_data[f'{loss_name}_loss'] = values
    
    # Add weights
    for i, weights_dict in enumerate(train_history['weights']):
        for loss_name, weight in weights_dict.items():
            if f'{loss_name}_weight' not in history_data:
                history_data[f'{loss_name}_weight'] = []
            history_data[f'{loss_name}_weight'].append(weight)
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(history_data)
    df.to_csv(save_path, index=False)
    print(f"Training history saved to {save_path}")


def save_evaluation_results(scores, labels, save_dir, experiment_name):
    """Save evaluation results and summary"""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    
    # Save raw results as JSON
    eval_results = {
        'scores': [float(score) for score in scores],
        'labels': [int(label) for label in labels],
        'num_samples': len(scores)
    }
    
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Create human-readable summary
    summary_path = os.path.join(save_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Evaluation Summary for {experiment_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(scores)}\n")
        f.write(f"Normal samples: {labels.count(0)}\n")
        f.write(f"Anomaly samples: {labels.count(1)}\n\n")
        
        # Calculate statistics
        scores_array = np.array(scores)
        labels_array = np.array(labels)
        
        normal_scores = scores_array[labels_array == 0]
        anomaly_scores = scores_array[labels_array == 1]
        
        f.write("Reconstruction Error Statistics:\n")
        f.write("-" * 30 + "\n")
        if len(normal_scores) > 0:
            f.write(f"Normal samples:\n")
            f.write(f"  Mean: {normal_scores.mean():.6f}\n")
            f.write(f"  Std:  {normal_scores.std():.6f}\n")
            f.write(f"  Min:  {normal_scores.min():.6f}\n")
            f.write(f"  Max:  {normal_scores.max():.6f}\n\n")
        
        if len(anomaly_scores) > 0:
            f.write(f"Anomaly samples:\n")
            f.write(f"  Mean: {anomaly_scores.mean():.6f}\n")
            f.write(f"  Std:  {anomaly_scores.std():.6f}\n")
            f.write(f"  Min:  {anomaly_scores.min():.6f}\n")
            f.write(f"  Max:  {anomaly_scores.max():.6f}\n\n")
        
        # Simple AUROC calculation if both classes present
        if len(normal_scores) > 0 and len(anomaly_scores) > 0:
            auroc = roc_auc_score(labels_array, scores_array)
            f.write(f"AUROC Score: {auroc:.4f}\n")
    
    print(f"Evaluation results saved to {results_path}")
    return eval_results


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