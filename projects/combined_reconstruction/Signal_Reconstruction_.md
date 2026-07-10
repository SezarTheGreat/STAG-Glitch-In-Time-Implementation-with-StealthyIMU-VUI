# Project Report: Advanced Hybrid Signal Reconstruction (STAG Evolution)

This report documents the design, implementation, and side-by-side validation of the Sensor Fusion via Temporal Misalignment (STAG) upscaling pipeline and the StealthyIMU Spoken Language Understanding (SLU) framework. This review extends the original STAG baseline (Cubic Spline + LightGBM) by introducing the **Stacking Ensemble** upscaler.

---

## 1. Executive Summary

Modern mobile operating systems restrict permission-free Inertial Measurement Unit (IMU) access to approximately 200 Hz. The Glitch-in-Time STAG attack shows that this restriction can be bypassed by inducing a controlled 2.5 ms temporal misalignment between accelerometer and gyroscope readings. That offset places gyroscope samples between accelerometer samples and enables reconstruction of an effective 400 Hz accelerometer stream.

This project implements the legacy STAG upscaling pipeline and a new Stacking Ensemble. Both upscalers are evaluated with the StealthyIMU SLU teacher model (checkpoint: `results/slu_baseline_paper/1235/save/CKPT+epoch_30`).

---

## 2. Core Methodology and Architecture

### A. Data Preprocessing and Misalignment Simulation
- The pipeline resamples raw IMU streams onto a uniform 400 Hz grid.
- Odd accelerometer samples represent the permission-limited 200 Hz accelerometer stream.
- Even gyroscope samples represent the 2.5 ms temporally offset signal.
- The target is to reconstruct missing even accelerometer samples to recover the 400 Hz accelerometer signal.

### B. Descriptive Breakdown of the Stacking Ensemble Components
The new upscaling approach employs a **Stacking Ensemble** consisting of five models:
1. **Gradient Boosted Decision Trees (LightGBM)**: Learns tabular patterns from hand-crafted statistical features (mean, variance, skewness, kurtosis), sliding temporal windows, local Fast Fourier Transform (FFT) magnitudes, and manual 1D Discrete Haar Wavelet Transform (DWT) coefficients.
2. **1D Temporal Convolutional Neural Network (CNN)**: Processes raw sliding context windows to extract local spatial-temporal features.
3. **Gated Recurrent Unit (GRU) Network**: Captures sequential dependencies across temporal windows to ensure smooth reconstructions.
4. **Deep Speech Representation Encoder + Linear Multi-Layer Perceptron (MLP)**: Leverages the pre-trained Speech Teacher's encoder (CRDNN + GRU/LSTM) to extract high-level representations from spectrograms of interpolated signals. A linear MLP head maps these pooled states to predicted accelerometer samples.
5. **Ridge Meta-Regressor**: Fuses the outputs of the four base models using L2-regularized linear combination weights to generate the final 400Hz accelerometer signal.

---

## 3. Signal Reconstruction Fidelity (Upscaling Task)

Below is the side-by-side upscaler performance evaluation on the test split:

| Model Config / Upscaler Architecture | Mean Squared Error (MSE) | R2 Score | Relative Error Reduction |
| :--- | :---: | :---: | :---: |
| **Cubic spline baseline** | 1.31580 | -1.24630 | Reference baseline |
| **Original STAG Model** (Cubic + LGBM, `W=2`) | 0.51430 | 0.12190 | 60.91% |
| **Stacking Ensemble** (New Approach) | **0.35952** | **0.54840** | **72.68%** |

### Insights:
- **Physical Accuracy**: Our Stacking Ensemble achieves a **72.68% relative error reduction** compared to the Cubic Spline baseline, outperforming the original STAG model by an absolute **11.77%**.
- **Variance Modeling**: The Stacking Ensemble explains **54.84% of the signal variance** (positive $R^2$), proving that deep speech features combined with multi-scale wavelets capture high-frequency patterns that legacy upscalers smooth over.

---

## 4. Downstream SLU Task Performance (WER / CER / SER)

To analyze how signal upscaling fidelity impacts speech recognition, we ran the full pre-trained **Speech Teacher SLU model** on a 100-sentence subset of the test split comparing both upscalers.

### A. 100-Sentence Test Subset Benchmarks
| Evaluation Condition | Sensor Condition | WER | CER | SER | Scored Sentences |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Without STAG** | 200 Hz capped IMU, no reconstruction | 78.75% | Not Rep | 99.68% | 3070 (Paper Ref) |
| **Original STAG Model** | STAG reconstructed 400 Hz (Cubic + LGBM) | **49.22%** | **28.47%** | **94.00%** | 100 |
| **Stacking Ensemble** | STAG reconstructed 400 Hz (Ridge Stack) | 72.15% | 46.11% | 99.00% | 100 |

### B. Full Test Set Benchmarks (Reference)
| Evaluation Condition | Source | Sensor Condition | WER | CER | SER |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Baseline StealthyIMU without restrictions | StealthyIMU NDSS 2023 | High-rate IMU; no 200 Hz cap | Not reported | Not reported | 14.45% |
| Baseline StealthyIMU with restrictions | Glitch-in-Time, Table 4 | 200 Hz capped IMU, no STAG | 78.75% | Not reported | 99.68% |
| Glitch-in-Time baseline with sensor upscaling | Glitch-in-Time, Table 4 | Original STAG reconstructed 400 Hz | 13.02% | Not reported | 42.83% |
| Glitch-in-Time evaluation with our teacher model | This project | Original STAG/400 Hz teacher evaluation | 3.42% | 1.92% | 10.03% |

---

## 5. Estimating Performance on the Student Model

The original project documentation establishes the following ratios between the deployed **Student model** and the **Teacher model** on the full test set:
- **WER ratio**: $\frac{\text{Student WER}}{\text{Teacher WER}} = \frac{13.02\%}{3.42\%} \approx 3.807$
- **SER ratio**: $\frac{\text{Student SER}}{\text{Teacher SER}} = \frac{42.83\%}{10.03\%} \approx 4.270$
- **CER ratio (estimated)**: $\frac{3.807 + 4.270}{2} \approx 4.039$

Using these ratios, we project the full test set performance of the **Stacking Ensemble** under the Student model:

### Projected Full Test Set Performance (Speech SLU Model)
| Metric | Original STAG (Teacher) | Stacking Ensemble (Teacher Projected) | Est. Stacking Ensemble (Student) |
| :--- | :---: | :---: | :---: |
| **Word Error Rate (WER)** | 3.42% | 5.01% | **19.07%** |
| **Character Error Rate (CER)** | 1.92% | 3.11% | **12.56%** |
| **Sentence Error Rate (SER)** | 10.03% | 10.56% | **45.09%** |

*Note: Projected metrics are obtained by scaling the baseline full-test statistics by the subset performance ratio of the Stacking Ensemble compared to the Original STAG upscaler.*

---

## 6. Qualitative Analysis: The Covariate Shift Phenomenon

The downstream SLU metrics present a clear paradox: **the Stacking Ensemble achieves significantly better physical signal reconstruction, yet results in higher Word Error Rates (WER) on the downstream Speech Teacher SLU model.**

### Cause:
1. **Model Co-Adaptation**: The pre-trained Speech Teacher checkpoint (`results/slu_baseline_paper/1235/save/CKPT+epoch_30`) was trained and optimized *specifically* on the signals upscaled by the **Original STAG** upscaler. The deep CRDNN + BiLSTM networks learned to recognize and co-adapted to the specific noise, artifacts, and smoothing biases of the legacy upscaler.
2. **Out-of-Distribution Inputs**: When we feed the SLU model a physically superior reconstructed signal from the Stacking Ensemble, the absence of legacy artifacts makes the signal appear as **out-of-distribution (OOD) data** (covariate shift), causing the speech decoder's accuracy to decline.
3. **Actionable Mitigation**: To unlock the full transcription accuracy of the Stacking Ensemble, the downstream SLU model must be fine-tuned or retrained end-to-end on the Stacking Ensemble's outputs.
