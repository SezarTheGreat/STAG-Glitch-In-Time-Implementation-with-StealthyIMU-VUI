# Experimentation Performance Summary

This document serves to summarize the progression of pipeline configurations tested across our replication and experimentation phases. It contrasts historical methodologies against the baseline metrics and recent signal-processing methods (such as spline interpolation, targeted high-pass filtering, and EMD IMF amplification).

## Performance Comparison Matrix

The table below breaks down the projected student model performance for each pipeline variant. 

| Pipeline Configuration | Projected Student WER (%) | Projected Student SER (%) | Projected Student SEER (%) | Status vs. Baselines |
| :--- | :--- | :--- | :--- | :--- |
| **StealthyIMU Old Method** | 78.75% | 99.68% | 68.42% | **[BASELINE 1]** Historical |
| **STAG Original Baseline** | 13.02% | 42.83% | 25.50% | **[BASELINE 2]** Paper Ref |
| **Post-Correction Butterworth** | **8.40%** | **27.03%** | **17.00%** | **[BASELINE 3]** Post-Filter |
| **STAG with Akima Splines** | **8.39%** | **27.00%** | *N/A* | Best Spline Variant |
| **Method 1 (Coherence Multiplier)** | 13.52% | 42.91% | 27.80% | Regressed |
| **Method 2 (Wiener Filtering)** | 12.33% | 42.91% | 24.30% | Beats Baseline 2 |
| **Method 3 (EMD IMF Amplification)** | 14.62% | 42.96% | 30.15% | Regressed |
| **Method 4 (Targeted High-Pass)** | 11.11% | 42.92% | 21.20% | Beats Baseline 2 |
| **Method 5 (Residual Correction)** | 14.78% | 42.96% | 30.80% | Regressed |
| **Peaking EQ (80-220 Hz, +6.0dB)** | 13.38% | 42.98% | 26.21% | Regressed |
| **Peaking EQ (80-220 Hz, +9.0dB)** | 13.39% | 42.97% | 26.23% | Regressed |
| **Day 13 Hybrid Model (VAD + Scaling)** | **22.02%** | **23.39%** | **21.29%** | Best SEER/SER, Regressed WER |
| **Day 13 AccEar cGAN + StealthyIMU (Physical)** | 15.40% (Est) | 95.90% | *TBD* | Domain Mismatch Baseline |

## Insights and Next Steps

- **Splines vs. Butterworth**: The Akima splines method provides marginal, almost indistinguishable improvements over the post-correction Butterworth filter (8.39% vs 8.40% WER), making both prime candidates for downstream signal stabilization.
- **AccEar cGAN Integration Performance (SER: 95.90%)**: The pipeline successfully executes end-to-end utilizing the exact physical dataset traces. The low performance (4.10% Intent Accuracy / 95.90% SER) stems from a **domain mismatch**. The checkpoint loaded from the workspace (`slu_kd_student`) is a student model trained on IMU motion spectrograms (`AccSpec` features), whereas the AccEar model outputs reconstructed *acoustic audio* Mel-spectrograms. Feeding audio characteristics into a motion-trained model results in baseline performance.
- **Future Resolution**: To obtain high VUI accuracy, the reconstructed audio must either be fed into a clean-audio-trained ASR/SLU model (the true Teacher model) or processed back into simulated IMU vibration characteristics.
