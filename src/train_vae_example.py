"""
VAE Training Example
====================

範例腳本展示如何使用 VAE 進行異常檢測訓練。
"""

import torch
import os
from pathlib import Path
from datetime import datetime

# Import modular components
from models import VariationalAutoencoder
from datasets import OpticalDataset
from losses import VAELoss, AnnealedVAELoss
from utils import (
    train_model,
    SyntheticAnomalyGenerator,
    create_optical_dataloader,
    evaluate_optical_model,
    get_device,
    create_experiment_name,
    create_experiment_directories,
    save_experiment_config,
    save_training_summary,
    save_training_history_csv,
    save_evaluation_results,
    plot_loss_curves
)
from visualization import AnomalyVisualizer


def main():
    """VAE 訓練主函數"""
    
    # 設定裝置
    device = get_device()
    print(f"Using device: {device}")
    
    # VAE 專用設定
    config = {
        'device': device,
        'architecture': 'vae',
        'batch_size': 16,
        'num_epochs': 50,
        'lr': 1e-3,
        'image_size': (176, 976),  # 或使用 (256, 256) 進行測試
        'use_synthetic_anomalies': True,
        'num_workers': 4,
        # VAE 專用損失設定
        'loss_config': {
            'vae': {
                'class': VAELoss,
                'weight': 1.0,
                'reconstruction_loss': 'mse',  # 或 'bce'
                'beta': 1.0  # β-VAE 參數，1.0 為標準 VAE
            }
        }
    }
    
    # 建立輸出目錄
    project_root = Path(__file__).parent.parent
    base_output_dir = project_root / 'out' / 'vae_experiments'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 建立實驗目錄
    experiment_name = f"vae_beta{config['loss_config']['vae']['beta']}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = create_experiment_directories(base_output_dir, experiment_name)
    
    # 儲存設定
    save_experiment_config(config, os.path.join(dirs['experiment'], 'training_config.json'))
    
    # 初始化模型
    model = VariationalAutoencoder(
        input_size=config['image_size'],
        latent_dim=128  # 可調整潛在空間維度
    )
    model = model.to(device)
    
    print(f"\n{'='*50}")
    print(f"Training VAE for Anomaly Detection")
    print(f"Image size: {config['image_size']}")
    print(f"Latent dimension: 128")
    print(f"β parameter: {config['loss_config']['vae']['beta']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output directory: {dirs['experiment']}")
    print(f"{'='*50}\n")
    
    # 建立訓練資料集
    train_dataset = OpticalDataset(
        '/home/yclai/vscode_project/Anomaly_Detection/OpticalDataset',
        split='train',
        transform=None,
        use_augmentation=True,
        synthetic_anomaly_generator=SyntheticAnomalyGenerator() if config['use_synthetic_anomalies'] else None
    )
    
    train_loader = create_optical_dataloader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    # 更新設定加入 checkpoint 目錄
    config_with_checkpoint = config.copy()
    config_with_checkpoint['checkpoint_dir'] = dirs['checkpoints']
    
    # 訓練模型
    print("Starting VAE training...")
    model, train_history = train_model(model, train_loader, config_with_checkpoint)
    
    # 儲存模型
    model_path = os.path.join(dirs['weights'], 'vae_final.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 儲存訓練歷史
    save_training_history_csv(train_history, os.path.join(dirs['history'], 'training_history.csv'))
    save_training_summary(train_history, config, experiment_name, 
                         os.path.join(dirs['experiment'], 'training_summary.txt'))
    
    # 繪製損失曲線
    plot_loss_curves(train_history, dirs['history'], experiment_name)
    
    # 評估模型
    print("\nEvaluating on test set...")
    test_dataset = OpticalDataset(
        '/home/yclai/vscode_project/Anomaly_Detection/OpticalDataset',
        split='test',
        transform=None
    )
    
    test_loader = create_optical_dataloader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # 設定視覺化
    vis_dir = os.path.join(dirs['experiment'], 'visualizations')
    visualizer = AnomalyVisualizer(save_dir=vis_dir)
    
    # 評估
    from losses import ModularLossManager
    loss_manager = ModularLossManager(config['loss_config'], device)
    scores, labels = evaluate_optical_model(model, test_loader, loss_manager, device)
    
    # 視覺化一些結果
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= 3:  # 只視覺化前 3 批
                break
                
            images = images.to(device)
            recon, mu, logvar = model(images)
            
            # 計算錯誤圖
            error_maps = torch.mean((images - recon) ** 2, dim=1)
            
            # 視覺化第一張圖
            visualizer.visualize_reconstruction(
                images[0],
                recon[0],
                error_maps[0].cpu().numpy(),
                save_name=f'vae_test_batch_{i}.png',
                show=False
            )
            
            # 額外：視覺化潛在空間的統計資訊
            print(f"\nBatch {i} latent space statistics:")
            print(f"  Mean μ: min={mu[0].min().item():.3f}, max={mu[0].max().item():.3f}, avg={mu[0].mean().item():.3f}")
            print(f"  Log variance: min={logvar[0].min().item():.3f}, max={logvar[0].max().item():.3f}, avg={logvar[0].mean().item():.3f}")
    
    # 儲存評估結果
    eval_results = save_evaluation_results(scores, labels, dirs['evaluation'], experiment_name)
    
    print(f"\n{'='*50}")
    print(f"VAE training completed!")
    print(f"Results saved in: {dirs['experiment']}")
    print(f"{'='*50}")
    
    # 展示如何從潛在空間採樣
    print("\nGenerating samples from latent space...")
    samples = model.sample(num_samples=4, device=device)
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        sample = samples[i, 0].cpu().numpy()
        ax.imshow(sample, cmap='gray')
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'vae_samples.png'))
    plt.close()
    print(f"Generated samples saved to {os.path.join(vis_dir, 'vae_samples.png')}")


def train_annealed_vae():
    """使用退火 β 排程訓練 VAE 的範例"""
    
    device = get_device()
    
    config = {
        'device': device,
        'architecture': 'vae',
        'batch_size': 16,
        'num_epochs': 100,
        'lr': 1e-3,
        'image_size': (256, 256),
        'use_synthetic_anomalies': True,
        'num_workers': 4,
        # 使用退火 VAE 損失
        'loss_config': {
            'vae': {
                'class': AnnealedVAELoss,
                'weight': 1.0,
                'reconstruction_loss': 'mse',
                'beta_start': 0.0,  # 開始時只關注重建
                'beta_end': 1.0,    # 最終達到標準 VAE
                'anneal_steps': 5000  # 退火步數
            }
        }
    }
    
    print("\nTraining VAE with annealed β schedule...")
    print(f"β will increase from {config['loss_config']['vae']['beta_start']} to {config['loss_config']['vae']['beta_end']} over {config['loss_config']['vae']['anneal_steps']} steps")
    
    # 其餘訓練流程與上面相同...


if __name__ == "__main__":
    # 執行標準 VAE 訓練
    main()
    
    # 如果要測試退火 VAE，可以取消註解：
    # train_annealed_vae()