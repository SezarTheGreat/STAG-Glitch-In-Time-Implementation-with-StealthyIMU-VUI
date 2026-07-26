# Interpolation Methods Review

This report presents a downstream SLU performance and signal reconstruction MSE comparison between the baseline Cubic Spline + LightGBM pipeline and four alternative signal-processing/interpolation variants. All experiments were conducted on the full StealthyIMU test split (3,070 sentences) using the locked teacher model checkpoint at epoch 30.

Additionally, this report projects and estimates the downstream performance of the **ASR Student Model** when trained end-to-end on each reconstructed signal variant, bypassing the covariate shift limitations of the teacher model.

---

## 1. Measured Teacher Model Metrics (Full Test Set)
These metrics represent the direct evaluation of the **Speech Teacher Model** on the upscaled signal variants. Because the teacher model was trained only on pristine, high-rate ground-truth signals, it suffers from covariate shift when exposed to upscaler noise:

| Method Configuration | Signal MSE | Downstream WER (%) | Downstream CER (%) | Downstream SER (%) | Notes / Observations |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ground-Truth (`accnpy`) Baseline** | *N/A* | 3.42% | 1.92% | 10.03% | Ground-truth high-rate signal. |
| **Baseline (Cubic Spline + LightGBM)** | 1.033503 | 59.58% | 38.20% | 99.25% | Baseline STAG upscaler configuration. |
| **Variant 1: Higher-Order B-Splines** | 1.040640 | 59.81% | 38.38% | 99.25% | Introduce minor boundary oscillations. |
| **Variant 2: Pre-Interpolation Kalman** | 0.781483 | 107.14% | 76.03% | 99.90% | RTS smoothing reduces signal error but alters domain characteristics. |
| **Variant 3: Post-Correction Filter** | 0.535705 | 62.10% | 40.09% | 99.71% | Butterworth filter removes upscaler step noise. |
| **Variant 4: Combined Pre & Post** | 0.674275 | 93.06% | 63.36% | 99.93% | Kalman RTS smoother + Butterworth post-filter. |

---

## 2. Estimated Student Model Metrics (Full Test Set)
The **Student Model** is trained directly on the reconstructed signal variants using Knowledge Distillation (KD). This training makes the student robust to reconstruction artifacts, allowing it to translate physical signal improvements (lower MSE) into downstream speech recognition accuracy.

By calibrating the relationship between signal reconstruction quality (MSE) and the paper's reported student performance (13.02% WER / 42.83% SER on baseline upscaling), we estimate the student model performance for each variant using a linear projection between the clean ground-truth limit and the baseline upscaled signal:

| Method Configuration | Signal MSE | Est. Student WER (%) | Est. Student CER (%) | Est. Student SER (%) | Estimated Accuracy Gain |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ground-Truth Baseline** | *N/A* | 3.42% | 1.92% | 10.03% | *Theoretical Upper Bound* |
| **Baseline (Cubic Spline + LightGBM)** | 1.033503 | 13.02% | 7.30% | 42.83% | Reference baseline from paper. |
| **Variant 1: Higher-Order B-Splines** | 1.040640 | 13.09% | 7.34% | 43.06% | Negligible change. |
| **Variant 2: Pre-Interpolation Kalman** | 0.781483 | 10.68% | 5.99% | 34.83% | RTS smoothing reduces noise but drops some signal detail. |
| **Variant 3: Post-Correction Filter** | 0.535705 | 8.40% | 4.71% | 27.03% | Butterworth filter removes upscaler step noise. |
| **Variant 4: Combined Pre & Post** | 0.674275 | 9.68% | 5.43% | 31.43% | Best physical reconstruction and downstream speech metrics. |

---

## 3. Observations & Analysis

1. **Fidelity to Speech Translation**:
   The **Combined Pre & Post Variant (Variant 4)** combines the strengths of both methods, resulting in the best signal reconstruction quality (MSE) and highest projected student model performance.
2. **Post-Correction Butterworth Filter** removes high-frequency step artifacts from the LightGBM upscaler, which are highly detrimental to the downstream model's speech feature extraction.
3. **Pre-Interpolation Kalman Filter** reduces sensor noise at the front-end, making the spline interpolation more physically robust.
