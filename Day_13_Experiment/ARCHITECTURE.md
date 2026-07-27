# Day 13 Architecture Scope & Design

## Hybrid Pipeline Architecture
The Day 13 Experiment fuses InertiEAR, STAG, and StealthyIMU methodologies into a 5-module pipeline.

### Module 1: Raw Data Capture and Intrinsic Denoising
- **Input**: 200 Hz Accelerometer and 200 Hz Gyroscope (StealthyIMU dataset format).
- **Processing**: 
  - Wiener Filtering (adaptive smoothing)
  - Stationary Noise Removal
  - DC Bias Removal
- **Goal**: Establish a mathematically clean baseline sensor trace.

### Module 2: Automatic Speech Segmentation (InertiEAR-inspired)
- **Processing**:
  - Temporary Linear Interpolation
  - Accelerometer–Gyroscope Alignment
  - Axis Multiplication (Energy calculation)
  - DC Bias Extraction
  - Otsu Thresholding
  - Speech Boundary Detection
- **Goal**: Autonomously mask out non-speech segments.

### Module 3: Device Independent Normalization
- **Processing**:
  - Dimension Reduction
  - Feature Normalization
  - Device Independent Scaling
- **Goal**: Generalize model inference across sensor hardware variance.

### Module 4: STAG Reconstruction Engine
- **Processing**:
  - Cubic Spline Interpolation (to approximate acoustic sampling rates)
  - LightGBM Prediction (first stage inference)
  - Second Stage LightGBM Fusion (residual/stacking correction)
  - 400 Hz Accelerometer Reconstruction
- **Goal**: Predict 400Hz acoustic equivalents from 200Hz preprocessed IMU signals.

### Module 5: Dual Branch Inference
- **Branch A: Seq2Seq (Unconstrained Speech)**
  - Embedding -> CNN Encoder -> BLSTM -> GRU Decoder with Attention.
- **Branch B: DenseNet (Targeted Vocabulary)**
  - Generates 244x244 Spectrograms -> DenseNet image classification.
- **Goal**: Parallel evaluation of both continuous unconstrained speech and rigid targeted commands.
