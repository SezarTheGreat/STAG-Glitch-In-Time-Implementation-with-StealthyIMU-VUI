# Day 13 Module Specifications

## Module 1: Raw Data Capture and Intrinsic Denoising
- **Responsibilities**: Load raw 200Hz IMU signals, apply Wiener filtering for adaptive smoothing, remove stationary background noise, and eliminate DC bias.
- **Expected Inputs**: Raw StealthyIMU `.csv` or tensor shapes `(N, 6, T)` (3 Accel, 3 Gyro).
- **Expected Outputs**: Cleaned signals of the same dimension `(N, 6, T)`.
- **Dependencies**: `scipy.signal` (Wiener).
- **Failure Modes**: Missing axis data, invalid sampling rate.

## Module 2: Automatic Speech Segmentation
- **Responsibilities**: Identify regions of actual speech using InertiEAR principles. Involves linear interpolation for timing alignment, axis multiplication for energy scaling, extracting a low-frequency envelope, and applying Otsu's method for dynamic boundary detection.
- **Expected Inputs**: Denoised signals `(N, 6, T)`.
- **Expected Outputs**: Masked signals or segmented intervals `(K, 6, T_sub)`.
- **Dependencies**: `common.interpolation.linear_interpolate`.
- **Failure Modes**: Low energy failing to trigger Otsu threshold; threshold collapsing to zero variance.

## Module 3: Device Independent Normalization
- **Responsibilities**: Standardize features to generalize across hardware (e.g., varying accelerometer sensitivities). Applies dimensionality reduction (e.g., PCA/LDA) and scaling.
- **Expected Inputs**: Segmented signals `(K, 6, T_sub)`.
- **Expected Outputs**: Normalized features `(K, D, T_sub)`.
- **Dependencies**: None.
- **Failure Modes**: Zero division on scaling if variance is 0.

## Module 4: STAG Reconstruction Engine
- **Responsibilities**: Upsample the 200Hz signal to a 400Hz acoustic-equivalent waveform using Cubic Spline Interpolation, followed by a two-stage LightGBM refinement to predict acoustic features.
- **Expected Inputs**: Normalized features `(K, D, T_sub)`.
- **Expected Outputs**: 400Hz reconstructed waveform `(K, 1, 2*T_sub)`.
- **Dependencies**: `common.interpolation.cubic_spline_interpolate`, `lightgbm`.
- **Failure Modes**: Nan generation during cubic spline inversion.

## Module 5: Dual Branch Inference
- **Responsibilities**: Process the reconstructed 400Hz acoustic signal for final downstream evaluation.
- **Branch A (Seq2Seq)**: For unconstrained speech mapping (Embedding -> CNN Encoder -> BLSTM -> Attention -> GRU Decoder).
- **Branch B (DenseNet)**: Converts signal to 244x244 spectrograms and evaluates closed-set targeted vocabulary.
- **Expected Inputs**: Reconstructed waveform `(K, 1, 2*T_sub)`.
- **Expected Outputs**: Transcriptions (Branch A) and Class Probabilities (Branch B).
- **Dependencies**: `torch`, `torchaudio`.
- **Failure Modes**: OOM errors during batch inference, padding mismatches.
