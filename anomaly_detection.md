需要注意的技術細節
1. 網路架構尺寸計算
在BaselineAutoencoder中，編碼器的尺寸變化需要精確計算：

976×176 → 488×88 → 244×44 → 122×22 → 61×11 → 31×6

由於976和176都不是2的完美倍數，連續的下採樣可能導致尺寸不匹配。建議在解碼器部分添加尺寸調整機制，確保最終輸出與輸入尺寸完全一致。
2. 合成異常生成邏輯
在訓練函數中，合成異常的處理邏輯需要更明確。當前的實現中，如果使用合成異常，資料集會返回(image, anomaly_mask)，但在訓練循環中的處理方式可能造成混淆。建議修改為：
pythonif config.get('use_synthetic_anomalies', False):
    # 資料集應該返回原始圖像和帶異常的圖像
    clean_images, anomaly_images, anomaly_masks = batch
    target = clean_images
    input_images = anomaly_images
else:
    images, _ = batch
    target = images
    input_images = images
3. 損失函數權重正規化
ModularLossFunction中的權重正規化是個好設計，但建議在初始化時檢查是否至少有一個損失函數被啟用，避免除以零的錯誤。