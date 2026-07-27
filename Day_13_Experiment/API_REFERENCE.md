# Day 13 API Reference

## Internal APIs to be Implemented

### Module 1: Preprocessing
- `apply_wiener(signal: np.ndarray, window_size: int) -> np.ndarray`
- `remove_dc_bias(signal: np.ndarray) -> np.ndarray`

### Module 2: Segmentation
- `compute_energy_envelope(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) -> np.ndarray`
- `apply_otsu_threshold(energy: np.ndarray) -> np.ndarray` (Returns boolean mask)

### Module 3: Normalization
- `device_independent_scaling(features: np.ndarray) -> np.ndarray`

### Module 4: STAG
- `lightgbm_stage1_predict(features: np.ndarray) -> np.ndarray`
- `lightgbm_stage2_fusion(preds: np.ndarray) -> np.ndarray`

### Module 5: Inference
- `class Seq2Seq(nn.Module)`: `forward(spectrogram)` -> `logits`
- `class TargetDenseNet(nn.Module)`: `forward(spectrogram_244)` -> `class_probs`
