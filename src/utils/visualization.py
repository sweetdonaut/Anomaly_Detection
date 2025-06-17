"""
Visualization Utilities
=======================

Functions for plotting and visualization.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import os


def plot_loss_curves(train_history, save_dir, experiment_name):
    """Plot and save training loss curves"""
    epochs = range(1, len(train_history['total_loss']) + 1)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Total loss and component losses
    ax1.plot(epochs, train_history['total_loss'], 'b-', label='Total Loss', linewidth=2)
    
    # Plot component losses
    colors = ['r', 'g', 'm', 'c', 'y', 'k']
    for i, (loss_name, values) in enumerate(train_history['component_losses'].items()):
        color = colors[i % len(colors)]
        ax1.plot(epochs, values, f'{color}--', label=f'{loss_name}', linewidth=1.5)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{experiment_name} - Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss weights over time
    for i, (loss_name, _) in enumerate(train_history['component_losses'].items()):
        weights = [w[loss_name] for w in train_history['weights']]
        color = colors[i % len(colors)]
        ax2.plot(epochs, weights, f'{color}-', label=f'{loss_name} weight', linewidth=1.5)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Weight')
    ax2.set_title('Loss Weights Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save a simpler plot with just the total loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_history['total_loss'], 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title(f'{experiment_name} - Total Loss')
    plt.grid(True, alpha=0.3)
    
    simple_plot_path = os.path.join(save_dir, 'total_loss_curve.png')
    plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to {plot_path}")


def plot_comparison_curves(all_histories, save_path, title="Loss Comparison"):
    """Plot comparison of multiple training histories"""
    plt.figure(figsize=(10, 6))
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    styles = ['-', '--', '-.', ':']
    
    for idx, (name, history) in enumerate(all_histories.items()):
        epochs = range(1, len(history['total_loss']) + 1)
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        plt.plot(epochs, history['total_loss'], f'{color}{style}', 
                label=name, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")


