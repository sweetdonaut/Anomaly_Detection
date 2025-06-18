import os
import torch
from torch.utils.data import Dataset
import tifffile
import numpy as np
from torchvision import transforms
from typing import Dict, List, Tuple, Optional

class OpticalDatasetTriplet(Dataset):
    """
    Dataset class for loading 4-channel TIFF images containing:
    - Channel 0: Target image (with modifications)
    - Channel 1: Reference image 1
    - Channel 2: Reference image 2
    - Channel 3: Mask (not used in training)
    """
    
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (976, 176),
        normalize: bool = True
    ):
        """
        Args:
            root_dir: 資料集根目錄
            mode: 'train' 或 'test'
            transform: 額外的資料增強轉換
            image_size: 影像尺寸 (H, W)
            normalize: 是否正規化到 [0, 1]
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.image_size = image_size
        self.normalize = normalize
        
        # 收集所有 TIFF 檔案
        if mode == 'test':
            # For test mode, collect files from all subdirectories
            test_path = os.path.join(root_dir, mode)
            self.file_list = []
            
            if os.path.exists(test_path):
                # Walk through all subdirectories
                for root, dirs, files in os.walk(test_path):
                    for file in files:
                        if file.endswith('.tiff'):
                            # Store relative path from root_dir
                            rel_path = os.path.relpath(os.path.join(root, file), root_dir)
                            self.file_list.append(rel_path)
            
            if not self.file_list:
                raise ValueError(f"No TIFF files found in {test_path}")
                
            self.file_list.sort()  # Sort for consistent ordering
        else:
            # For train mode, keep original behavior
            self.data_path = os.path.join(root_dir, mode, 'good')
            if not os.path.exists(self.data_path):
                raise ValueError(f"Data path not found: {self.data_path}")
            
            self.file_list = [os.path.join(mode, 'good', f) for f in os.listdir(self.data_path) if f.endswith('.tiff')]
            if not self.file_list:
                raise ValueError(f"No TIFF files found in {self.data_path}")
        
        print(f"Found {len(self.file_list)} files in {mode} set")
        
        # 基本轉換
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
            - 'target': Target image tensor [1, H, W]
            - 'reference1': Reference image 1 tensor [1, H, W]
            - 'reference2': Reference image 2 tensor [1, H, W]
            - 'references': Stacked references [2, H, W]
            - 'filename': 檔案名稱
        """
        # 載入 4 通道 TIFF 影像
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        tiff_data = tifffile.imread(file_path)
        
        # 確保資料格式正確
        if tiff_data.shape[0] != 4:
            raise ValueError(f"Expected 4 channels, got {tiff_data.shape[0]} for {file_path}")
        
        # 分離各通道
        target = tiff_data[0]      # Channel 0: Target
        ref1 = tiff_data[1]        # Channel 1: Reference 1
        ref2 = tiff_data[2]        # Channel 2: Reference 2
        # mask = tiff_data[3]      # Channel 3: Mask (not used)
        
        # 轉換為浮點數並正規化
        if self.normalize:
            target = target.astype(np.float32) / 255.0
            ref1 = ref1.astype(np.float32) / 255.0
            ref2 = ref2.astype(np.float32) / 255.0
        else:
            target = target.astype(np.float32)
            ref1 = ref1.astype(np.float32)
            ref2 = ref2.astype(np.float32)
        
        # 轉換為 PyTorch tensors (添加 channel 維度)
        target = torch.from_numpy(target).unsqueeze(0)  # [1, H, W]
        ref1 = torch.from_numpy(ref1).unsqueeze(0)      # [1, H, W]
        ref2 = torch.from_numpy(ref2).unsqueeze(0)      # [1, H, W]
        
        # 堆疊 references
        references = torch.cat([ref1, ref2], dim=0)     # [2, H, W]
        
        # 應用額外的轉換（如果有）
        if self.transform:
            # 對所有影像應用相同的轉換以保持一致性
            seed = torch.initial_seed()
            
            torch.manual_seed(seed)
            target = self.transform(target)
            
            torch.manual_seed(seed)
            ref1 = self.transform(ref1)
            
            torch.manual_seed(seed)
            ref2 = self.transform(ref2)
            
            references = torch.cat([ref1, ref2], dim=0)
        
        return {
            'target': target,
            'reference1': ref1,
            'reference2': ref2,
            'references': references,
            'filename': self.file_list[idx]
        }
    
    def get_sample_batch(self, batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """
        獲取一個樣本批次用於快速測試
        """
        indices = np.random.choice(len(self), min(batch_size, len(self)), replace=False)
        batch = [self[i] for i in indices]
        
        return {
            'target': torch.stack([b['target'] for b in batch]),
            'reference1': torch.stack([b['reference1'] for b in batch]),
            'reference2': torch.stack([b['reference2'] for b in batch]),
            'references': torch.stack([b['references'] for b in batch]),
            'filenames': [b['filename'] for b in batch]
        }


def create_triplet_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (976, 176),
    augment_train: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    建立訓練和測試的 DataLoader
    """
    # 訓練資料的轉換
    train_transform = None
    if augment_train:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # 可以加入其他增強，但要注意保持 target 和 references 的一致性
        ])
    
    # 建立 datasets
    train_dataset = OpticalDatasetTriplet(
        root_dir=root_dir,
        mode='train',
        transform=train_transform,
        image_size=image_size
    )
    
    test_dataset = OpticalDatasetTriplet(
        root_dir=root_dir,
        mode='test',
        transform=None,
        image_size=image_size
    )
    
    # 建立 dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # 測試 dataset
    dataset = OpticalDatasetTriplet(
        root_dir="../../triplet_dataset",
        mode='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 取得一個樣本
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Reference1 shape: {sample['reference1'].shape}")
    print(f"Reference2 shape: {sample['reference2'].shape}")
    print(f"References shape: {sample['references'].shape}")
    print(f"Filename: {sample['filename']}")
    
    # 測試批次
    batch = dataset.get_sample_batch(batch_size=2)
    print(f"\nBatch target shape: {batch['target'].shape}")
    print(f"Batch references shape: {batch['references'].shape}")