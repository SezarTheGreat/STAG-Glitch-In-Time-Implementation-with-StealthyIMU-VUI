# Pre-Interpolation Kalman Filter Pipeline Analysis

This document describes the signal processing and machine learning pipeline of the **Variant 2 Pre-Interpolation Kalman Filter (RTS Smoother)** compared to the original STAG upscaler pipeline (modeled after Figure 6 of the original paper).

---

## 1. Visual Comparison of the Pipelines

The diagram below contrasts the original baseline STAG pipeline and the new Pre-Interpolation Kalman Filter pipeline.

```mermaid
graph TD
    %% Styling
    classDef default fill:#1e1e2e,stroke:#cdd6f4,stroke-width:1px,color:#cdd6f4;
    classDef highlight fill:#89b4fa,stroke:#89b4fa,stroke-width:2px,color:#11111b;
    classDef stage fill:#a6e3a1,stroke:#a6e3a1,stroke-width:1px,color:#11111b;

    subgraph Original_STAG ["Original STAG Pipeline (Paper Figure 6 Baseline)"]
        A[Odd Acc Samples 200Hz] --> B[Cubic Spline Interpolation]
        B -->|Interpolated Even Acc| D[Context Window Feature Extractor]
        C[Even Gyro Samples 200Hz] --> D
        D -->|Feature Matrix W=2| E[LightGBM Model]
        E -->|Predicted Even Acc| F[Interleave Odd & Even]
        A -->|True Odd Acc| F
        F -->|Upscaled 400Hz Signal| G[Resample to 500Hz]
        G -->|500Hz Acc Waveform| H[Extract Spectrogram Features]
        H -->|Speech Features| I[Speech SLU Model]
    end

    subgraph Variant_2 ["Variant 2: Pre-Interpolation Kalman Filter Pipeline"]
        A2[Odd Acc Samples 200Hz] --> J2["RTS Smoother (Denoising)"]:::highlight
        C2[Even Gyro Samples 200Hz] --> K2["RTS Smoother (Denoising)"]:::highlight
        
        J2 -->|Cleaned Odd Acc| B2[Cubic Spline Interpolation]
        B2 -->|Interpolated Even Acc| D2[Context Window Feature Extractor]
        K2 -->|Cleaned Even Gyro| D2
        
        D2 -->|Feature Matrix W=2| E2[LightGBM Model]
        E2 -->|Predicted Even Acc| F2[Interleave Odd & Even]
        J2 -->|Cleaned Odd Acc| F2
        
        F2 -->|Upscaled 400Hz Signal| G2[Resample to 500Hz]
        G2 -->|500Hz Acc Waveform| H2[Extract Spectrogram Features]
        H2 -->|Speech Features| I2[Speech SLU Model]
    end
```

---

## 2. Key Differences in the Architecture Pipeline

| Pipeline Stage | Original STAG Baseline (Figure 6) | Variant 2: Pre-Interpolation Kalman Filter |
| :--- | :--- | :--- |
| **Noise Filtering** | **None.** The raw 200 Hz sensor readings go directly to interpolation and features. | **Applied First.** RTS state-space smoothers run on both 200 Hz streams to remove sensor white noise. |
| **Interpolation** | Cubic spline of Odd Acc (200Hz) to Even grid (200Hz). | Cubic spline of the **cleaned** Odd Acc (200Hz) to Even grid (200Hz). |
| **Features Input** | Features are extracted from raw, noisy Gyro and interpolated Acc. | Features are extracted from **cleaned** Gyro and interpolated Acc. |
| **Model Inference** | LightGBM predicts corrected even acc using raw features. | LightGBM predicts corrected even acc using denoised features. |
| **Interleaving** | True odd and predicted even samples are interleaved. | **Cleaned** odd and predicted even samples are interleaved. |
| **Downstream Pipeline** | Reconstructed signal resampled to 500 Hz for features. | Reconstructed signal resampled to 500 Hz for features. |

---

## 3. Why the Pre-Interpolation Kalman Filter is Used

1.  **Removing Sensor White Noise**:
    Physical micro-electro-mechanical system (MEMS) sensors in mobile phones contain thermal and electrical noise. By utilizing a constant-velocity kinematic state-space model, the Kalman filter estimates the true physical motion, filtering out this noise before any interpolation takes place.
2.  **Improving Spline Trajectories**:
    Splines are highly sensitive to outlier noise. Interpolating a noisy 200 Hz signal generates severe trajectory oscillations. Denoising the signal first ensures that the cubic spline computes a physically realistic trajectory.
3.  **Preventing Error Propagation**:
    If the features fed to LightGBM contain raw sensor noise, the model's predictions inherit this noise. By cleaning the inputs, we prevent error propagation through the regression model.
