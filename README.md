# 無監督學習異常檢測研究專案

本專案專注於研究無監督學習方法在異常檢測領域的應用，目標是探索和比較不同的模型架構、損失函數和訓練策略，以找出最佳的異常檢測解決方案。

## 專案目標

- 🔍 **研究無監督學習方法**：探索各種無監督學習技術在異常檢測中的應用
- 🏗️ **模型架構優化**：實驗不同的自編碼器架構和深度學習模型
- 📊 **損失函數比較**：評估不同損失函數對異常檢測性能的影響
- 🎯 **訓練策略研究**：開發和測試最佳的訓練策略和超參數配置

## 當前實作

### 模型架構

#### GrayscaleAnomalyAutoencoder
- **架構類型**：U-Net 風格的編碼器-解碼器
- **輸入格式**：單通道灰階影像（1008×176 解析度）
- **特色**：
  - 跳躍連接以保留空間資訊
  - 批次正規化和 ReLU 激活
  - 針對灰階影像優化的通道配置
  - 總參數量：約 15M 參數

### 損失函數

#### EnhancedGrayscaleCombinedLoss
- **組合損失**：結合 L2 (MSE) 和多尺度 SSIM
- **動態權重調整**：訓練過程中自動調整損失權重
- **多尺度 SSIM**：適用於不同尺度的結構相似性評估
- **非對稱視窗**：針對寬高比影像優化的 SSIM 計算

### 訓練策略

- **學習率調度**：餘弦退火學習率調度器
- **批次大小**：16（針對灰階影像記憶體優化）
- **訓練輪數**：100 epochs
- **優化器**：Adam 優化器
- **數據增強**：標準化處理（mean=0.5, std=0.5）

## 資料集

使用 **MVTec 資料集**進行實驗：
- **格式**：灰階影像
- **類別**：目前支援 'grid' 類別（可擴展至其他類別）
- **分割**：訓練集用於模型訓練，測試集用於異常檢測評估

## 專案結構

```
Anomaly_Detection/
├── README.md                           # 專案說明文件
├── CLAUDE.md                          # Claude Code 指導文件
└── MVTec_unsupervised/                # 無監督學習實作
    └── AE_single_channel.py           # 單通道自編碼器實作
```

## 使用方法

### 環境需求

```bash
pip install torch torchvision numpy pillow matplotlib scipy tqdm pathlib
```

### 訓練模型

```python
python MVTec_unsupervised/AE_single_channel.py
```

### 推理使用

#### 檔案基礎推理
```python
from MVTec_unsupervised.AE_single_channel import inference_grayscale

# 對單張影像進行異常檢測
heatmap, reconstruction = inference_grayscale(
    model_path='grid_grayscale_autoencoder.pth',
    image_path='test_image.png',
    category='grid'
)
```

#### 張量基礎推理
```python
from MVTec_unsupervised.AE_single_channel import inference_grayscale_tensor

# 對張量輸入進行異常檢測
heatmap, reconstruction = inference_grayscale_tensor(
    model_path='grid_grayscale_autoencoder.pth',
    image_tensor=your_tensor,
    category='grid'
)
```

## 核心特色

### 1. 進階損失函數
- **多尺度 SSIM**：評估不同尺度的結構相似性
- **動態權重調整**：隨訓練進度自動調整 L2 和 SSIM 權重
- **非對稱視窗**：適應寬高比影像的 SSIM 計算

### 2. 異常檢測
- **像素級重建誤差**：計算原始影像與重建影像的差異
- **高斯平滑**：提升異常熱力圖的視覺化效果
- **自動正規化**：確保異常分數在 [0,1] 範圍內

### 3. 視覺化功能
- **三聯圖顯示**：原始影像、重建影像、異常熱力圖
- **熱力圖彩色映射**：使用 'hot' 色彩映射突出異常區域
- **自動保存結果**：生成高解析度的檢測結果圖

## 配置參數

在 `main()` 函數中可調整的關鍵參數：

```python
device = 'cuda'                # 計算設備
batch_size = 16               # 批次大小
num_epochs = 100              # 訓練輪數
image_size = 1024             # 影像大小
dataset_path = '/path/to/data'  # 資料集路徑
categories = ['grid']         # 訓練類別
```

## 未來發展方向

### 模型架構研究
- [ ] 變分自編碼器 (VAE)
- [ ] 生成對抗網路 (GAN)
- [ ] Vision Transformer 架構
- [ ] 記憶增強網路

### 損失函數探索
- [ ] 感知損失 (Perceptual Loss)
- [ ] 特徵匹配損失
- [ ] 對抗損失
- [ ] 自定義異常敏感損失

### 訓練策略優化
- [ ] 課程學習
- [ ] 自監督預訓練
- [ ] 增量學習
- [ ] 多任務學習

### 評估指標
- [ ] AUC-ROC 曲線
- [ ] AUC-PR 曲線
- [ ] 像素級和物體級評估
- [ ] 速度和記憶體效率分析

## 貢獻

歡迎對本專案提出建議和改進：
1. Fork 專案
2. 建立特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 授權

本專案採用 MIT 授權條款 - 詳見 LICENSE 文件

## 聯絡資訊

如有任何問題或建議，請隨時聯絡專案維護者。