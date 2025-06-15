"""
Simple Inference Script for Trained Models
==========================================

Load trained models and generate reconstruction visualizations.
All parameters are hardcoded for easy execution.
"""

import torch
import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Import necessary components
from models import BaselineAutoencoder, EnhancedAutoencoder, CompactAutoencoder, CompactUNetAutoencoder
from datasets import OpticalDataset
from losses import ModularLossManager
from utils import get_device
from visualization import AnomalyVisualizer


# ===== CONFIGURATION - MODIFY THESE VALUES =====
DATE_FOLDER = '2025_0615_2320'  # Date folder to load model from
EXPERIMENT_NAME = 'compact_mse'  # Experiment name (e.g., 'compact_mse', 'baseline_mse')
NUM_SAMPLES = 20  # Number of samples to visualize
BATCH_SIZE = 8  # Batch size for inference
DEFECT_TYPES = ['broken', 'bent']  # List of defect types to test (e.g., ['broken', 'bent']) or None for all
# Available types: 'broken', 'bent', 'glue', 'thread', 'metal_contamination', 'good'
# ===============================================


def load_model_from_experiment(experiment_path, device):
    """Load model from experiment directory"""
    
    # Load training config
    config_path = os.path.join(experiment_path, 'training_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Fix loss_config if it was saved as strings
    if 'loss_config' in config:
        from losses import MSELoss, SSIMLoss, MultiScaleSSIMLoss, SobelGradientLoss, FocalFrequencyLoss
        
        loss_classes = {
            'MSELoss': MSELoss,
            'SSIMLoss': SSIMLoss,
            'MultiScaleSSIMLoss': MultiScaleSSIMLoss,
            'SobelGradientLoss': SobelGradientLoss,
            'FocalFrequencyLoss': FocalFrequencyLoss
        }
        
        # Convert string class names back to actual classes
        for loss_name, loss_info in config['loss_config'].items():
            if 'class' in loss_info and isinstance(loss_info['class'], str):
                class_name = loss_info['class']
                if class_name in loss_classes:
                    loss_info['class'] = loss_classes[class_name]
    
    # Extract architecture from experiment name
    experiment_name = os.path.basename(experiment_path)
    architecture = experiment_name.split('_')[0]
    
    # Initialize model based on architecture
    if architecture == 'baseline':
        model = BaselineAutoencoder(input_size=tuple(config['image_size']))
    elif architecture == 'enhanced':
        model = EnhancedAutoencoder()
    elif architecture == 'compact':
        model = CompactAutoencoder(input_size=tuple(config['image_size']))
    elif architecture == 'compact-unet':
        model = CompactUNetAutoencoder(input_size=tuple(config['image_size']))
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Load weights
    weights_path = os.path.join(experiment_path, 'weights', 'final_model.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, config


def create_filtered_dataloader(defect_types=None):
    """Create dataloader with optional defect type filtering"""
    
    # Load full test dataset
    test_dataset = OpticalDataset(
        '/home/yclai/vscode_project/Anomaly_Detection/OpticalDataset',
        split='test',
        transform=None
    )
    
    if defect_types is None:
        # Use all test data
        filtered_dataset = test_dataset
        filtered_paths = test_dataset.image_paths
    else:
        # Ensure it's a list
        if isinstance(defect_types, str):
            defect_types = [defect_types]
        
        # Filter for specific defect types
        filtered_indices = []
        filtered_paths = []
        type_counts = {dt: 0 for dt in defect_types}
        
        for idx in range(len(test_dataset)):
            img_path = test_dataset.image_paths[idx]
            # Check if the path contains any of the defect types
            for defect_type in defect_types:
                if f'/{defect_type}/' in img_path:
                    filtered_indices.append(idx)
                    filtered_paths.append(img_path)
                    type_counts[defect_type] += 1
                    break
        
        if not filtered_indices:
            raise ValueError(f"No images found for defect types: {defect_types}")
        
        # Create subset
        filtered_dataset = torch.utils.data.Subset(test_dataset, filtered_indices)
        print(f"Filtered to {len(filtered_indices)} images:")
        for dt, count in type_counts.items():
            print(f"  {dt}: {count} images")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        filtered_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    return dataloader, test_dataset, filtered_paths


def visualize_reconstructions(model, dataloader, device, save_dir, visualizer, image_paths):
    """Generate and save individual reconstruction visualizations"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    sample_count = 0
    all_scores = []
    defect_info = []
    defect_counts = {}  # Track count per defect type
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if sample_count >= NUM_SAMPLES:
                break
            
            # Handle both subset and full dataset returns
            if isinstance(batch_data, tuple):
                images, labels = batch_data
            else:
                images = batch_data
                labels = torch.zeros(images.shape[0])
                
            images = images.to(device)
            reconstructions = model(images)
            
            # Calculate pixel-wise MSE for error maps
            errors = torch.mean((images - reconstructions) ** 2, dim=1)
            
            # Process each image in batch
            for i in range(images.shape[0]):
                if sample_count >= NUM_SAMPLES:
                    break
                
                # Get image path and extract defect type and filename
                global_idx = batch_idx * BATCH_SIZE + i
                if global_idx < len(image_paths):
                    img_path = image_paths[global_idx]
                    # Extract defect type and filename from path
                    path_parts = img_path.split('/')
                    defect_type = 'unknown'
                    original_filename = 'unknown.png'
                    
                    for j, part in enumerate(path_parts):
                        if part == 'test' and j + 1 < len(path_parts):
                            defect_type = path_parts[j + 1]
                            if j + 2 < len(path_parts):
                                original_filename = path_parts[j + 2]
                                # Change extension to .png
                                filename_base = original_filename.rsplit('.', 1)[0]
                                original_filename = f"{filename_base}.png"
                            break
                else:
                    defect_type = 'unknown'
                    original_filename = 'unknown.png'
                    img_path = 'unknown'
                
                # Create subdirectory for defect type
                defect_dir = os.path.join(save_dir, defect_type)
                os.makedirs(defect_dir, exist_ok=True)
                
                # Update visualizer save directory temporarily
                original_save_dir = visualizer.save_dir
                visualizer.save_dir = defect_dir
                
                # Calculate MSE score for this sample
                mse_score = errors[i].mean().item()
                all_scores.append(mse_score)
                defect_info.append((sample_count + 1, defect_type, mse_score, original_filename))
                
                # Track count per defect type
                if defect_type not in defect_counts:
                    defect_counts[defect_type] = 0
                defect_counts[defect_type] += 1
                
                # Use visualizer to create the visualization with original filename
                visualizer.visualize_reconstruction(
                    images[i],
                    reconstructions[i],
                    errors[i].cpu().numpy(),
                    save_name=original_filename,
                    show=False
                )
                
                # Restore original save directory
                visualizer.save_dir = original_save_dir
                
                print(f"Sample {sample_count+1} ({defect_type}/{original_filename}): MSE = {mse_score:.6f}")
                sample_count += 1
    
    print(f"\nSaved {sample_count} reconstruction visualizations to {save_dir}")
    print(f"Average MSE: {np.mean(all_scores):.6f} ± {np.std(all_scores):.6f}")
    
    # Print summary by defect type
    print("\nSummary by defect type:")
    defect_scores = {}
    for _, defect_type, score, _ in defect_info:
        if defect_type not in defect_scores:
            defect_scores[defect_type] = []
        defect_scores[defect_type].append(score)
    
    for defect_type, scores in defect_scores.items():
        print(f"  {defect_type}: {len(scores)} samples, avg MSE = {np.mean(scores):.6f}")
    
    return all_scores, defect_info


def calculate_all_anomaly_scores(model, dataloader, loss_manager, device):
    """Calculate anomaly scores for all samples in dataloader"""
    
    all_scores = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_data in dataloader:
            # Handle both subset and full dataset returns
            if isinstance(batch_data, tuple):
                images, labels = batch_data
            else:
                images = batch_data
                labels = torch.zeros(images.shape[0])
                
            images = images.to(device)
            reconstructions = model(images)
            
            # Calculate loss for each sample
            batch_scores = []
            for i in range(images.shape[0]):
                img = images[i:i+1]
                rec = reconstructions[i:i+1]
                
                # Calculate total loss using loss manager
                loss_dict = loss_manager(rec, img)
                batch_scores.append(loss_dict['total'].item())
            
            all_scores.extend(batch_scores)
            all_labels.extend(labels.tolist())
    
    return np.array(all_scores), np.array(all_labels)


def main():
    """Main inference function"""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    out_dir = project_root / 'out'
    date_dir = out_dir / DATE_FOLDER
    experiment_dir = date_dir / EXPERIMENT_NAME
    
    if not experiment_dir.exists():
        print(f"Error: Experiment directory {experiment_dir} not found!")
        return
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create inference output directory
    inference_dir = out_dir / f"{DATE_FOLDER}_inference"
    if DEFECT_TYPES:
        # Create folder name with defect types
        defect_types_str = '_'.join(DEFECT_TYPES) if isinstance(DEFECT_TYPES, list) else DEFECT_TYPES
        exp_output_dir = inference_dir / f"{EXPERIMENT_NAME}_{defect_types_str}"
    else:
        exp_output_dir = inference_dir / EXPERIMENT_NAME
    
    os.makedirs(exp_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running inference for: {EXPERIMENT_NAME}")
    if DEFECT_TYPES:
        print(f"Filtering for defect types: {DEFECT_TYPES}")
    print(f"Results will be saved to: {exp_output_dir}")
    print(f"{'='*60}\n")
    
    try:
        # Load model and config
        model, config = load_model_from_experiment(str(experiment_dir), device)
        print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create filtered dataloader
        dataloader, full_dataset, image_paths = create_filtered_dataloader(DEFECT_TYPES)
        
        # Setup visualization
        vis_dir = os.path.join(exp_output_dir, 'visualizations')
        visualizer = AnomalyVisualizer(save_dir=vis_dir)
        
        # Generate visualizations
        print("\nGenerating reconstruction visualizations...")
        sample_scores, defect_info = visualize_reconstructions(
            model, dataloader, device, vis_dir, visualizer, image_paths
        )
        
        # Calculate anomaly scores for all samples
        print("\nCalculating anomaly scores for all samples...")
        loss_manager = ModularLossManager(config['loss_config'], device)
        all_scores, all_labels = calculate_all_anomaly_scores(
            model, dataloader, loss_manager, device
        )
        
        # Save scores
        scores_path = exp_output_dir / 'anomaly_scores.npz'
        np.savez(scores_path, scores=all_scores, labels=all_labels)
        print(f"Saved anomaly scores to {scores_path}")
        
        # Save statistics
        stats_path = exp_output_dir / 'score_statistics.txt'
        with open(stats_path, 'w') as f:
            f.write("Anomaly Score Statistics\n")
            f.write("=" * 50 + "\n\n")
            if DEFECT_TYPES:
                f.write(f"Defect types: {DEFECT_TYPES}\n")
            f.write(f"Total samples: {len(all_scores)}\n")
            f.write(f"Mean score: {np.mean(all_scores):.6f}\n")
            f.write(f"Std score: {np.std(all_scores):.6f}\n")
            f.write(f"Min score: {np.min(all_scores):.6f}\n")
            f.write(f"Max score: {np.max(all_scores):.6f}\n")
            f.write(f"Median score: {np.median(all_scores):.6f}\n")
            
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            f.write("\nPercentiles:\n")
            for p in percentiles:
                if len(all_scores) > 0:
                    f.write(f"  {p}%: {np.percentile(all_scores, p):.6f}\n")
            
            f.write("\n\nVisualized samples:\n")
            for sample_id, defect_type, score, filename in defect_info:
                f.write(f"  Sample {sample_id} ({defect_type}/{filename}): MSE = {score:.6f}\n")
        
        print(f"\nStatistics saved to {stats_path}")
        
    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()