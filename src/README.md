# 模組化異常檢測系統

此專案已完成模組化重構，將原本的單一檔案拆分為多個功能模組。

## 專案結構

```
src/
├── __init__.py
├── main.py                 # 主訓練程式
├── losses/                 # 損失函數模組
│   ├── __init__.py
│   ├── base.py            # 基礎損失類別
│   ├── mse.py             # MSE 損失
│   ├── ssim.py            # SSIM 損失
│   ├── ms_ssim.py         # Multi-Scale SSIM 損失
│   ├── sobel.py           # Sobel 梯度損失
│   ├── focal_frequency.py # Focal Frequency 損失
│   └── manager.py         # 模組化損失管理器
├── models/                 # 模型架構
│   ├── __init__.py
│   ├── baseline.py        # 基礎自編碼器
│   └── enhanced.py        # 增強型自編碼器（U-Net風格）
├── datasets/              # 資料集載入器
│   ├── __init__.py
│   └── mvtec.py          # MVTec AD 資料集
├── utils/                 # 工具函數
│   ├── __init__.py
│   ├── synthetic_anomaly.py  # 合成異常生成器
│   ├── latent_analyzer.py    # 潛在空間分析器
│   └── training.py           # 訓練相關函數
├── visualization/         # 視覺化工具
│   ├── __init__.py
│   └── visualizer.py     # 異常視覺化器
├── test/                  # 測試檔案
│   └── test_modular.py   # 模組化測試
└── backup/               # 原始檔案備份
    ├── anomaly_detection_v2.py
    ├── anomaly_detection_v3.py
    └── anomaly_detection_v4.py
```

## 使用方式

### 1. 執行主訓練程式
```bash
cd src
python main.py
```

### 2. 執行測試程式
```bash
cd src/test
python test_modular.py
```

### 3. 自訂訓練配置

在 `main.py` 中修改配置：

```python
config = {
    'architecture': 'enhanced',  # 或 'baseline'
    'loss_config': {
        'mse': {'class': MSELoss, 'weight': 0.5},
        'ssim': {'class': SSIMLoss, 'weight': 0.5}
    }
}
```

### 4. 使用個別模組

```python
# 導入特定損失函數
from losses import MSELoss, SSIMLoss

# 導入模型
from models import BaselineAutoencoder

# 導入工具
from utils import SyntheticAnomalyGenerator
```

## 主要改進

1. **模組化設計**：每個功能都有獨立的模組，便於維護和擴展
2. **清晰的介面**：統一的 API 設計，所有損失函數繼承自 `BaseLoss`
3. **易於擴展**：可輕鬆添加新的損失函數或模型架構
4. **保持相容性**：功能與原始 v4 版本完全相同

## 新增功能建議

若要新增自訂損失函數：

1. 在 `losses/` 目錄下創建新檔案
2. 繼承 `BaseLoss` 類別
3. 實作 `forward` 方法
4. 在 `losses/__init__.py` 中導入

範例：
```python
# losses/custom_loss.py
from .base import BaseLoss

class CustomLoss(BaseLoss):
    def forward(self, pred, target):
        # 實作您的損失計算
        return loss_value
```