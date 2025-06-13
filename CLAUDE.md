# CLAUDE.md

此檔案為 Claude Code (claude.ai/code) 在此代碼庫中工作時的指導文件。

## 語言偏好
請使用繁體中文回應所有對話和說明。

## 專案概述

這是一個使用卷積自編碼器在 MVTec 資料集上進行無監督學習的異常檢測系統。專案專注於使用 U-Net 風格的自編碼器架構結合增強型損失函數（MSE 和 MS-SSIM）來檢測灰階影像中的異常。

## 架構

主要組件包括：

- **GrayscaleAnomalyAutoencoder**：U-Net 風格的編碼器-解碼器，具有跳躍連接，專為單通道灰階影像設計（1008x176 解析度）
- **EnhancedGrayscaleCombinedLoss**：結合 L2 和多尺度 SSIM 的進階損失函數，具有訓練期間的動態權重調整
- **GrayscaleAnomalyDetector**：推理類別，使用像素級重建誤差生成異常熱力圖
- **MVTecGrayscaleDataset**：用於載入 MVTec 影像作為灰階的自定義 PyTorch 資料集

## 主要特色

- 針對寬高比影像（1008x176）優化的非對稱 SSIM 視窗
- 在訓練期間逐漸增加 L2 權重的動態損失權重調度
- 用於更好異常熱力圖視覺化的高斯平滑
- 支援檔案路徑和張量基礎的推理

## 常用指令

### 訓練
```python
python MVTec_unsupervised/AE_single_channel.py
```

### 模型使用
主腳本處理訓練並將模型儲存為 `{category}_grayscale_autoencoder.pth`。推理使用方式：

```python
# 檔案基礎推理
heatmap, recon = inference_grayscale('model.pth', 'image.png', 'category')

# 張量基礎推理
heatmap, recon = inference_grayscale_tensor('model.pth', tensor, 'category')
```

## 配置設定

`main()` 中的關鍵參數：
- `batch_size = 16`（由於灰階效率可以增加）
- `num_epochs = 100`
- `image_size = 1024`
- 資料集路徑：`/Users/laiyongcheng/Desktop/autoencoder/`
- 類別：`['grid']`（可擴展至其他 MVTec 類別）

## 相依套件

- PyTorch（建議使用 CUDA 支援）
- torchvision
- numpy
- PIL (Pillow)
- matplotlib
- scipy
- tqdm
- pathlib