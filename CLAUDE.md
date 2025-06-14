# CLAUDE.md

此檔案為 Claude Code (claude.ai/code) 在此代碼庫中工作時的指導文件。

## 語言偏好
請使用繁體中文回應所有對話和說明。

## 專案概述

這是一個基於重建模型的無監督異常檢測系統，專為工業缺陷檢測設計。系統僅使用正常影像訓練自編碼器，透過分析重建誤差來檢測異常。專案實作了模組化架構，支援多種損失函數、合成異常生成和完整的評估指標。

## 專案結構（模組化版本）

```
src/
├── main.py                 # 主訓練程式
├── losses/                 # 損失函數模組
│   ├── base.py            # 基礎損失類別
│   ├── mse.py             # MSE 損失
│   ├── ssim.py            # SSIM 損失
│   ├── ms_ssim.py         # Multi-Scale SSIM 損失
│   ├── sobel.py           # Sobel 梯度損失
│   ├── focal_frequency.py # Focal Frequency 損失
│   └── manager.py         # 模組化損失管理器
├── models/                 # 模型架構
│   ├── baseline.py        # 基礎自編碼器
│   └── enhanced.py        # 增強型自編碼器（U-Net風格）
├── datasets/              # 資料集載入器
│   └── mvtec.py          # MVTec AD 資料集
├── utils/                 # 工具函數
│   ├── synthetic_anomaly.py  # 合成異常生成器
│   ├── latent_analyzer.py    # 潛在空間分析器
│   └── training.py           # 訓練相關函數
├── visualization/         # 視覺化工具
│   └── visualizer.py     # 異常視覺化器
├── test/                  # 測試檔案
│   └── test_modular.py   # 模組化測試
└── backup/               # 原始檔案備份（v1-v4）
```

## 架構設計

### 核心組件

1. **模組化損失函數框架 (ModularLossManager)**
   - **MSE Loss**：像素級重建精度
   - **SSIM Loss**：結構相似性保留，包含詳細實作與參數驗證
   - **Multi-Scale SSIM Loss**：多尺度結構相似性，同時捕捉局部細節與全局結構
   - **Focal Frequency Loss**：動態聚焦於難以重建的頻率成分
   - **Sobel Gradient Loss**：邊緣資訊保留
   - 支援靈活的權重配置和組合測試
   - 所有損失函數繼承自 `BaseLoss`，統一權重管理

2. **雙重網路架構**
   - **BaselineAutoencoder**：無跳躍連接的標準自編碼器，強制資訊壓縮
   - **EnhancedAutoencoder**：具有 U-Net 風格跳躍連接，精確缺陷定位
   - 兩種架構都支援可變輸入尺寸（預設 1024×1024）

3. **合成異常生成器 (SyntheticAnomalyGenerator)**
   - 生成亮點/暗點異常，外觀真實
   - 橢圓形狀，邊緣平滑過渡
   - 可配置大小和強度變化
   - 支援批次處理

4. **潛在空間分析器 (LatentSpaceAnalyzer)**
   - 多層特徵提取（來自編碼器中間層）
   - L2 距離計算高層語義差異
   - 無需預訓練網路的領域特定特徵學習

5. **異常視覺化器 (AnomalyVisualizer)**
   - 視覺化原始影像、重建影像和異常熱圖
   - 異常分數儲存功能
   - 批次處理支援

## 主要特色

- **影像尺寸**：支援任意尺寸，預設 1024×1024 像素（灰階單通道）
- **模組化設計**：便於實驗不同配置和擴展功能
- **訓練模式**：支援有/無合成異常的訓練
- **保守資料增強**：縮放係數 0.95-1.05
- **綜合異常評分**：結合重建誤差和潛在空間分析
- **單通道優化**：所有組件皆已驗證支援單通道影像處理
- **全英文程式碼**：註解和文檔使用英文，適合生產環境

## 常用指令

### 訓練模組化版本
```bash
cd src
python main.py
```

### 執行測試
```bash
cd src/test
python test_modular.py
```

### 訓練舊版模型（備份檔案）
```bash
# v4 版本（最新的單檔案版本）
python src/backup/anomaly_detection_v4.py

# v3 版本
python src/backup/anomaly_detection_v3.py
```

## 配置設定

主要配置參數（在 `src/main.py` 中）：
```python
config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 16,
    'num_epochs': 100,
    'lr': 1e-3,
    'image_size': (1024, 1024),  # 支援任意尺寸
    'architecture': 'enhanced',    # 'baseline' 或 'enhanced'
    'use_synthetic_anomalies': True,
    'loss_config': {
        # MSE Loss: 基本像素級重建
        'mse': {
            'class': MSELoss,
            'weight': 0.3
        },
        # SSIM Loss: 結構相似性保留
        'ssim': {
            'class': SSIMLoss,
            'weight': 0.3,
            'params': {'window_size': 11, 'sigma': 1.5}
        },
        # Focal Frequency Loss: 動態聚焦難重建區域
        'focal_freq': {
            'class': FocalFrequencyLoss,
            'weight': 0.2,
            'params': {'alpha': 1.0, 'patch_factor': 1}
        },
        # Sobel Gradient Loss: 邊緣保留
        'sobel': {
            'class': SobelGradientLoss,
            'weight': 0.2
        }
    },
    'save_path': './models'
}
```

### 範例配置

1. **僅使用 MSE + SSIM（快速訓練）**
```python
'loss_config': {
    'mse': {'class': MSELoss, 'weight': 0.5},
    'ssim': {'class': SSIMLoss, 'weight': 0.5, 'params': {'window_size': 11}}
}
```

2. **強調頻率域（適合紋理影像）**
```python
'loss_config': {
    'mse': {'class': MSELoss, 'weight': 0.2},
    'focal_freq': {'class': FocalFrequencyLoss, 'weight': 0.5, 'params': {'alpha': 2.0}},
    'sobel': {'class': SobelGradientLoss, 'weight': 0.3}
}
```

3. **多尺度 SSIM 重點**
```python
'loss_config': {
    'mse': {'class': MSELoss, 'weight': 0.3},
    'ms_ssim': {'class': MultiScaleSSIMLoss, 'weight': 0.5, 'params': {'num_scales': 3}},
    'sobel': {'class': SobelGradientLoss, 'weight': 0.2}
}
```

## 擴展指南

### 新增自訂損失函數

1. 在 `src/losses/` 建立新檔案：
```python
# src/losses/custom_loss.py
from .base import BaseLoss

class CustomLoss(BaseLoss):
    def __init__(self, weight=1.0, custom_param=0.5):
        super().__init__(weight)
        self.custom_param = custom_param
    
    def forward(self, pred, target):
        # 實作損失計算
        loss = ...
        return loss
```

2. 在 `src/losses/__init__.py` 中導入：
```python
from .custom_loss import CustomLoss
```

3. 在配置中使用：
```python
'loss_config': {
    'custom': {'class': CustomLoss, 'weight': 0.3, 'params': {'custom_param': 0.7}}
}
```

### 新增模型架構

1. 在 `src/models/` 建立新檔案
2. 確保模型有 `forward()` 方法
3. 在 `src/models/__init__.py` 中導入

## 實驗設計流程

1. **階段一：基準建立**
   - 使用標準自編碼器測試各種損失組合
   - 評估重建品質和異常檢測效能

2. **階段二：合成異常整合**
   - 實作真實缺陷生成
   - 比較有/無合成異常的效能

3. **階段三：架構比較**
   - 評估標準與跳躍連接變體
   - 分析不同異常尺度的表現

4. **階段四：損失函數優化**
   - 測試個別損失組件
   - 探索加權組合

## 最新更新（模組化版本）

1. **完整模組化重構**
   - 將單一檔案拆分為功能明確的模組
   - 保持與 v4 版本完全相同的功能
   - 提升代碼可維護性和可擴展性

2. **改進的專案結構**
   - 清晰的目錄組織
   - 統一的 API 設計
   - 獨立的測試框架

3. **支援彈性配置**
   - 可變影像尺寸支援
   - 動態 CPU 工作執行緒配置
   - 易於新增自訂組件

4. **增強的 MS-SSIM 實作**
   - 簡化版本提升穩定性
   - 修正梯度流問題
   - 支援所有影像尺寸

## 相依套件

- PyTorch（建議使用 CUDA 支援）
- torchvision
- numpy
- PIL (Pillow)
- matplotlib
- scipy
- tqdm
- pathlib
- scikit-learn（用於評估指標）
- opencv-python（用於合成異常生成）
- typing（用於型別提示）
- multiprocessing（用於優化資料載入）