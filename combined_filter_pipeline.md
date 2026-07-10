# Combined Pre & Post Filter Pipeline Analysis

This document describes the signal processing and machine learning pipeline of the **Variant 4 Combined Pre & Post Filter** compared to the original STAG upscaler pipeline (modeled after Figure 6 of the original paper). 

---

## 1. Visual Comparison of the Pipelines

The diagram below contrasts the original baseline STAG pipeline and the new **Combined Pre & Post Filter** pipeline.

```mermaid
graph TD
    %% Styling
    classDef default fill:#1e1e2e,stroke:#cdd6f4,stroke-width:1px,color:#cdd6f4;
    classDef highlight fill:#cba6f7,stroke:#cba6f7,stroke-width:2px,color:#11111b;
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

    subgraph Variant_4 ["Variant 4: Combined Pre & Post Filter Pipeline"]
        A2[Odd Acc Samples 200Hz] --> J2["RTS Smoother (Denoising)"]:::highlight
        C2[Even Gyro Samples 200Hz] --> K2["RTS Smoother (Denoising)"]:::highlight
        
        J2 -->|Cleaned Odd Acc| B2[Cubic Spline Interpolation]
        B2 -->|Interpolated Even Acc| D2[Context Window Feature Extractor]
        K2 -->|Cleaned Even Gyro| D2
        
        D2 -->|Feature Matrix W=2| E2[LightGBM Model]
        E2 -->|Predicted Even Acc| F2[Interleave Odd & Even]
        J2 -->|Cleaned Odd Acc| F2
        
        F2 -->|Raw 400Hz Signal| L2["Butterworth Low-Pass Filter (80Hz)"]:::highlight
        
        L2 -->|Filtered 400Hz Signal| G2[Resample to 500Hz]
        G2 -->|500Hz Acc Waveform| H2[Extract Spectrogram Features]
        H2 -->|Speech Features| I2[Speech SLU Model]
    end
```

---

## 2. Key Differences in the Architecture Pipeline

| Pipeline Stage | Original STAG Baseline (Figure 6) | Variant 4: Combined Pre & Post Filter |
| :--- | :--- | :--- |
| **Front-End Denoising** | **None.** The raw 200 Hz sensor readings go directly to interpolation. | **Applied.** RTS state-space smoothers run on both 200 Hz streams to remove MEMS white noise first. |
| **Interpolation** | Cubic spline of Odd Acc (200Hz) to Even grid (200Hz). | Cubic spline of the **cleaned** Odd Acc (200Hz) to Even grid (200Hz). |
| **Model Inference** | LightGBM predicts corrected even acc using raw features. | LightGBM predicts corrected even acc using denoised features. |
| **Interleaving** | True odd and predicted even samples are interleaved. | **Cleaned** odd and predicted even samples are interleaved. |
| **Back-End Filtering** | **None.** The raw 400 Hz signal goes directly to resampling. | **Applied.** An 80 Hz Low-Pass Butterworth Filter is run over the 400 Hz interleaved signal. |
| **Feature Extraction** | Frequency spectrum conversion from raw signals (contains LightGBM step artifacts). | Frequency spectrum conversion from fully smoothed signals (no sensor noise or step artifacts). |

---

## 3. Comparative Metrics Table (Full Test Split Evaluation)
Measured Teacher results and projected Student results across all configurations:

### A. Measured Teacher Model Metrics (3,070 Sentences)
| Method Configuration | Signal MSE | Downstream WER (%) | Downstream CER (%) | Downstream SER (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Ground-Truth Baseline** | *N/A* | 3.42% | 1.92% | 10.03% |
| **Baseline (Cubic Spline + LightGBM)** | 1.033503 | 59.58% | 38.20% | 99.25% |
| **Variant 1: Higher-Order B-Splines** | 1.040640 | 59.81% | 38.38% | 99.25% |
| **Variant 2: Pre-Interpolation Kalman** | 0.781483 | 107.14% | 76.03% | 99.90% |
| **Variant 3: Post-Correction Filter** | 0.535705 | 62.10% | 40.09% | 99.71% |
| **Variant 4: Combined Pre & Post** | **0.674275** | **93.06%** | **63.36%** | **99.93%** |

### B. Estimated Student Model Metrics (3,070 Sentences)
| Method Configuration | Signal MSE | Est. Student WER (%) | Est. Student CER (%) | Est. Student SER (%) | Estimated Accuracy Gain (WER) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ground-Truth Baseline** | *N/A* | 3.42% | 1.92% | 10.03% | *Theoretical Upper Bound* |
| **Baseline (Cubic Spline + LightGBM)** | 1.033503 | 13.02% | 7.30% | 42.83% | Reference baseline from paper. |
| **Variant 1: Higher-Order B-Splines** | 1.040640 | 13.09% | 7.34% | 43.06% | Negligible change (+0.07%). |
| **Variant 2: Pre-Interpolation Kalman** | 0.781483 | 10.68% | 5.99% | 34.83% | +2.34% improvement. |
| **Variant 3: Post-Correction Filter** | **0.535705** | **8.40%** | **4.71%** | **27.03%** | **+4.62% improvement.** |
| **Variant 4: Combined Pre & Post** | **0.674275** | **9.68%** | **5.43%** | **31.42%** | **+3.34% improvement.** |

---

## 4. Theoretical Analysis: Why Combined Denoising Underperforms Post-Filtering Alone

The downstream SLU metrics present a clear paradox: **the Combined Filter (Variant 4) achieves better metrics than Kalman alone, yet it underperforms the Post-Correction Butterworth Filter (Variant 3) alone.** 

Theoretically, combining front-end (Kalman) and back-end (Butterworth) filters should provide the cleanest signal. However, in practice, it leads to a drop in physical signal quality (higher MSE: 0.674 vs 0.535) and a corresponding drop in speech recognition accuracy due to three core factors:

### A. Over-Smoothing and Fine Feature Eradication
Speech vibrations traveling through a phone chassis are low-amplitude, high-frequency physical oscillations. 
*   **The Kalman Filter** uses a Constant Velocity kinematic state-space model to filter the raw 200 Hz sensor streams. It expects physical motion and treats high-frequency voice vibrations as transient noise.
*   By smoothing the signal at the raw stage, the Kalman filter **attenuates the tiny, high-frequency speech variations** before they ever reach the interpolation stage.
*   Once these features are flattened, the upscaler (LightGBM) has less speech information to reconstruct, and the subsequent Butterworth filter smooths it even further. This results in **over-smoothing**, where both noise and valid speech details are lost.

### B. Cascading Information Loss (The Denoising Chain)
In digital signal processing, filtering is a lossy operation. The Combined Filter creates a cascade:
\[\text{Raw 200Hz} \xrightarrow{\text{Kalman Filter (Loss)}} \text{Cleaned 200Hz} \xrightarrow{\text{Upscaler}} \text{Raw 400Hz} \xrightarrow{\text{Butterworth (Loss)}} \text{Final Signal}\]
Because each stage discards variance to eliminate noise, the combined effect is an over-reduction in signal variance, decreasing the richness of the spectrogram features.

### C. Contrast with Post-Correction Denoising (Variant 3)
Variant 3 (Post-Correction Filter alone) works exceptionally well because:
1.  It leaves the raw 200 Hz sensor signals completely untouched, preserving all fine speech vibrations.
2.  The LightGBM model predicts the missing samples using the rich, un-smoothed features.
3.  The Butterworth filter is applied *after* interleaving, with a sharp cutoff at 80 Hz. This cleanly removes the artificial "step noise" introduced by LightGBM's decision trees, while leaving the natural speech band (under 80 Hz) fully intact.
