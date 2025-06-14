# 無監督異常檢測系統 (Unsupervised Anomaly Detection System)

基於深度學習自編碼器的工業缺陷檢測系統，專為 MVTec 資料集設計，支援無標籤的異常檢測。

## 📋 目錄

- [系統概述](#系統概述)
- [運作流程圖](#運作流程圖)
- [主要特色](#主要特色)
- [安裝需求](#安裝需求)
- [快速開始](#快速開始)
- [系統架構](#系統架構)
- [使用方式](#使用方式)
- [參數配置](#參數配置)
- [輸出結果](#輸出結果)

## 🔍 系統概述

本系統使用自編碼器架構進行無監督異常檢測，僅需正常樣本進行訓練。透過分析重建誤差和潛在空間特徵，能夠有效檢測出異常區域。

## 🔄 運作流程圖

### 訓練階段 (Training Phase)

```
┌─────────────────────────────────────────────────────────────────┐
│                         資料準備階段                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   正常圖片資料   │ ───► │    資料增強      │ ───► │  合成異常生成    │
│  (976×176 灰階)  │      │  • 旋轉 ±5°     │      │  • 亮點/暗點     │
└─────────────────┘      │  • 縮放 95-105%  │      │  • 10×10 像素   │
                         └─────────────────┘      │  • 強度 0.2-0.4  │
                                                  └─────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         模型訓練階段                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
           ┌──────────────┐        ┌──────────────┐
           │ Baseline AE  │        │ Enhanced AE  │
           │ (無跳躍連接) │        │ (U-Net 架構) │
           └──────────────┘        └──────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    ┌─────────────────────┐
                    │    編碼器 (Encoder)  │
                    │  976×176 → 30×5     │
                    └─────────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │    瓶頸層           │
                    │  (Bottleneck)       │
                    └─────────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │   解碼器 (Decoder)  │
                    │   30×5 → 976×176    │
                    └─────────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │   模組化損失函數     │
                    │  • MSE (30%)        │
                    │  • SSIM (30%)       │
                    │  • Focal Freq (20%)│
                    │  • Sobel Edge (20%)│
                    └─────────────────────┘
```

### 推理階段 (Inference Phase)

```
┌─────────────────────────────────────────────────────────────────┐
│                         推理階段                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │    輸入圖片         │
                    │  (可能含異常)       │
                    └─────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
           ┌──────────────┐        ┌──────────────┐
           │  重建誤差     │        │ 潛在空間分析 │
           │  計算        │        │ (L2 距離)    │
           └──────────────┘        └──────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    ┌─────────────────────┐
                    │    異常分數計算     │
                    │ Score = MSE + 0.5×L2│
                    └─────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
           ┌──────────────┐        ┌──────────────┐
           │  異常熱力圖   │        │   視覺化輸出  │
           │  (Heatmap)   │        │  • 原圖      │
           └──────────────┘        │  • 重建圖    │
                                   │  • 差異圖    │
                                   └──────────────┘
```

## ✨ 主要特色

### 1. **無需標籤訓練**
- 僅使用正常樣本進行訓練
- 自動學習正常模式的特徵表示

### 2. **雙重網路架構**
- **Baseline Autoencoder**: 標準架構，強制資訊壓縮
- **Enhanced Autoencoder**: U-Net 風格，精確缺陷定位

### 3. **合成異常生成**
- 訓練時自動生成亮點/暗點缺陷
- 提升模型對異常的敏感度

### 4. **模組化損失函數**
- MSE: 像素級重建精度
- SSIM: 結構相似性保留
- Focal Frequency Loss: 頻率域特徵
- Sobel Gradient Loss: 邊緣資訊保留

### 5. **智慧資源管理**
- 自動偵測 CPU 核心數
- 動態調整資料載入器的工作進程數

## 📦 安裝需求

```bash
# 基本套件
pip install torch torchvision
pip install numpy pillow matplotlib
pip install tqdm pathlib

# 可選套件（用於合成異常生成）
pip install opencv-python

# 建議使用 CUDA 支援的 PyTorch 版本以加速訓練
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 快速開始

### 1. 準備資料
確保您的資料結構如下：
```
/path/to/dataset/
├── category_name/
│   ├── train/
│   │   └── good/
│   │       ├── 000.png
│   │       ├── 001.png
│   │       └── ...
│   └── test/
│       ├── good/
│       └── defect_type/
```

### 2. 執行訓練
```python
python MVTec_unsupervised/anomaly_detection_v2.py
```

### 3. 查看結果
訓練完成後，結果將儲存在：
- 模型檔案：`./models/{category}_final_model.pth`
- 視覺化結果：`./models/visualizations_{category}/`

## 🏗️ 系統架構

### 網路尺寸變化
```
輸入: 976×176 (灰階)
├── 編碼器路徑:
│   ├── 976×176 → 488×88 (Conv 3×3, stride 2)
│   ├── 488×88 → 244×44
│   ├── 244×44 → 122×22
│   ├── 122×22 → 61×11
│   └── 61×11 → 30×5 (最終編碼)
│
└── 解碼器路徑:
    ├── 30×5 → 61×11 (ConvTranspose)
    ├── 61×11 → 122×22
    ├── 122×22 → 244×44
    ├── 244×44 → 488×88
    └── 488×88 → 976×176 (輸出)
```

### 激活函數
- 使用 SiLU (Swish) 取代傳統 ReLU
- 提供更平滑的梯度流
- 避免死神經元問題

## 💻 使用方式

### 基本使用
```python
from anomaly_detection_v2 import *

# 載入訓練好的模型
model = EnhancedAutoencoder()
model.load_state_dict(torch.load('grid_final_model.pth'))
model.eval()

# 進行推理
transform = transforms.Compose([
    transforms.Resize((976, 176)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 載入圖片
image = Image.open('test_image.png').convert('L')
image_tensor = transform(image).unsqueeze(0)

# 生成異常熱力圖
with torch.no_grad():
    recon = model(image_tensor)
    diff = torch.abs(image_tensor - recon)
    heatmap = diff[0, 0].numpy()
```

### 自定義訓練
```python
# 修改配置
config = {
    'batch_size': 8,           # 批次大小
    'num_epochs': 50,          # 訓練週期
    'lr': 5e-4,                # 學習率
    'architecture': 'baseline', # 或 'enhanced'
    'use_synthetic_anomalies': False,  # 關閉合成異常
}
```

## ⚙️ 參數配置

### 主要配置參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `batch_size` | 16 | 訓練批次大小 |
| `num_epochs` | 100 | 訓練週期數 |
| `lr` | 1e-3 | 學習率 |
| `image_size` | (976, 176) | 輸入影像尺寸 |
| `architecture` | 'enhanced' | 網路架構選擇 |
| `use_synthetic_anomalies` | True | 是否使用合成異常 |

### 損失函數權重

| 組件 | 權重 | 功能 |
|------|------|------|
| MSE | 0.3 | 像素級精確度 |
| SSIM | 0.3 | 結構相似性 |
| Focal Frequency | 0.2 | 頻率域特徵 |
| Sobel Gradient | 0.2 | 邊緣保留 |

## 📊 輸出結果

### 1. 模型檔案
- `{category}_final_model.pth`: 最終訓練模型
- `checkpoint_epoch_{n}.pth`: 每 10 個 epoch 的檢查點

### 2. 視覺化結果
每個測試樣本包含三張圖：
- **原始圖片**: 輸入的測試影像
- **重建圖片**: 模型重建的結果
- **異常熱力圖**: 顯示異常區域（紅色=高異常分數）

### 3. 異常分數統計
- 平均異常分數
- 最大異常分數
- 最小異常分數

## 🔧 進階功能

### 1. 批次推理
```python
# 對整個資料夾進行異常檢測
def batch_inference(model, folder_path, transform):
    results = []
    for img_path in Path(folder_path).glob('*.png'):
        # 處理每張圖片
        score = compute_anomaly_score(model, img_path, transform)
        results.append((img_path, score))
    return results
```

### 2. 自定義異常閾值
```python
# 基於訓練資料統計設定閾值
normal_scores = compute_normal_scores(model, train_loader)
threshold = np.mean(normal_scores) + 3 * np.std(normal_scores)
```

## 📝 注意事項

1. **記憶體需求**: Enhanced 架構需要較多 GPU 記憶體
2. **訓練時間**: 100 epochs 約需 1-2 小時（取決於 GPU）
3. **影像格式**: 系統預期灰階 PNG 影像
4. **正規化**: 影像使用 mean=0.5, std=0.5 正規化

## 🤝 貢獻指南

歡迎提交 Issue 或 Pull Request 來改進系統！

## 📄 授權

本專案採用 MIT 授權條款。