"""
File I/O Utilities
==================

Functions for saving and loading experiment data.
"""

import json
import os
import pandas as pd
import numpy as np


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
        
    
    print(f"Evaluation results saved to {results_path}")
    return eval_results


def load_experiment_config(config_path):
    """Load experiment configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_training_history(csv_path):
    """Load training history from CSV file"""
    df = pd.read_csv(csv_path)
    return df