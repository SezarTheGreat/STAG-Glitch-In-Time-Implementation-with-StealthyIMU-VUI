# Post-Correction Butterworth Filter Pipeline Analysis

This document describes the signal processing and machine learning pipeline of the **Variant 3 Post-Correction Filter** compared to the original STAG upscaler pipeline (modeled after Figure 6 of the original paper). 

---

## 1. Visual Comparison of the Pipelines

The diagram below contrasts the original baseline STAG pipeline and the new Post-Correction Butterworth Filter pipeline.

```mermaid
graph TD
    %% Styling
    classDef default fill:#1e1e2e,stroke:#cdd6f4,stroke-width:1px,color:#cdd6f4;
    classDef highlight fill:#f38ba8,stroke:#f38ba8,stroke-width:2px,color:#11111b;
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

    subgraph Variant_3 ["Variant 3: Post-Correction Filter Pipeline"]
        A2[Odd Acc Samples 200Hz] --> B2[Cubic Spline Interpolation]
        B2 -->|Interpolated Even Acc| D2[Context Window Feature Extractor]
        C2[Even Gyro Samples 200Hz] --> D2
        D2 -->|Feature Matrix W=2| E2[LightGBM Model]
        E2 -->|Predicted Even Acc| F2[Interleave Odd & Even]
        A2 -->|True Odd Acc| F2
        
        %% Highlighted Step
        F2 -->|Raw 400Hz Signal| J2["Butterworth Low-Pass Filter (80Hz)"]:::highlight
        
        J2 -->|Filtered 400Hz Signal| G2[Resample to 500Hz]
        G2 -->|500Hz Acc Waveform| H2[Extract Spectrogram Features]
        H2 -->|Speech Features| I2[Speech SLU Model]
    end
```

---

## 2. Key Differences in the Architecture Pipeline

| Pipeline Stage | Original STAG Baseline (Figure 6) | Variant 3: Post-Correction Butterworth Filter |
| :--- | :--- | :--- |
| **Interpolation** | Cubic spline of Odd Acc (200Hz) to Even grid (200Hz). | Cubic spline of Odd Acc (200Hz) to Even grid (200Hz). |
| **Model Inference** | LightGBM predicts the corrected even accelerometer values. | LightGBM predicts the corrected even accelerometer values. |
| **Interleaving** | True odd and predicted even samples are interleaved to form a **raw 400 Hz signal**. | True odd and predicted even samples are interleaved to form a **raw 400 Hz signal**. |
| **Signal Filtering** | **None.** The raw 400 Hz signal goes directly to resampling. | **Applied.** An 80 Hz Low-Pass Butterworth Filter is run over the 400 Hz interleaved signal. |
| **Resampling** | Resamples the raw 400 Hz upscaled signal to 500 Hz. | Resamples the **filtered** 400 Hz upscaled signal to 500 Hz. |
| **Feature Extraction** | Frequency spectrum conversion from raw signals (contains LightGBM step artifacts). | Frequency spectrum conversion from smoothed signals (artifacts are filtered out). |

---

## 3. Why the Post-Filter Improves Performance

1.  **Elimination of LightGBM "Step" Noise**:
    LightGBM is a decision-tree-based regressor. Trees output piecewise constant values, which introduce sudden stair-step jumps (high-frequency noise) in the interleaved time-domain signal. The Butterworth filter smooths out these jagged jumps.
2.  **Frequency Matching**:
    Human speech vibrations carried through solid phone chassis rarely exceed **80 Hz**. Any energy in the reconstructed 400 Hz signal above 80 Hz is typically upscaling noise or numerical artifacts. A low-pass filter with a cutoff at 80 Hz acts as a perfect noise gate, removing this out-of-band noise.
3.  **Cleaner Spectrograms**:
    Because the high-frequency step artifacts are removed before the STFT (Short-Time Fourier Transform) is computed, the resulting spectrogram features are much cleaner and contain far fewer acoustic distortions when fed to the downstream SpeechBrain SLU model.
