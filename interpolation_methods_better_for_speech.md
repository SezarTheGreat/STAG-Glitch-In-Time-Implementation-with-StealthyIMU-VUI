# Interpolation Methods for SpeechBrain SLU Models

In the STAG framework, we reconstruct high-rate (400 Hz) accelerometer/gyroscope signals from lower-rate (200 Hz) streams. For downstream spoken language understanding (SLU) models like SpeechBrain's CRDNN, the signal representation is transformed into the frequency domain using a Short-Time Fourier Transform (STFT) or log-mel filterbank. 

Standard geometric interpolation methods (such as cubic, B-spline, or Akima splines) can introduce spectral artifacts. Below, we analyze why alternative digital signal processing (DSP) interpolation techniques are more suited for this downstream acoustic pipeline.

---

## 1. The Limitation of Splines for Acoustic Features

Spline methods are designed to produce **geometric smoothness** (continuously differentiable curves). They optimize for physical smoothness (minimizing acceleration/jerk) by fitting piecewise polynomials. 

However, speech-induced phone vibration is a high-frequency, oscillatory, band-limited signal. Splines suffer from the following drawbacks in this domain:
- **High-Frequency Spectral Leakage**: Spline functions are piecewise polynomials; their higher-order derivatives are discontinuous at boundaries. In the frequency domain, these discontinuities act as high-frequency step changes or "kinks" that manifest as spurious high-frequency noise in the spectrogram.
- **Out-of-Band Artifacts**: Splines do not enforce a strict frequency limit. They can introduce frequencies higher than the Nyquist limit of the original 200 Hz stream, causing spectral aliasing when transformed via STFT.
- **Acoustic Distortions**: Because SpeechBrain models are trained on real audio (or high-rate sensors converted directly to spectrograms), they are sensitive to these synthetic high-frequency polynomial shapes, leading to a severe **covariate shift** (manifested by high Word Error Rates).

---

## 2. DSP-Aligned Alternative Interpolation Methods

For SpeechBrain and other STFT-based downstream speech/audio models, the interpolation method should preserve the frequency-domain integrity of the signal. The following methods are more inline with this goal:

### A. Whittaker-Shannon (Sinc) Interpolation
According to the Nyquist-Shannon sampling theorem, a band-limited continuous-time signal can be perfectly reconstructed from its discrete samples using sinc interpolation:
\[x(t) = \sum_{n=-\infty}^{\infty} x(nT) \cdot \text{sinc}\left(\frac{t - nT}{T}\right)\]
- **Downstream Benefit**: Acting as an ideal low-pass brick-wall filter, it ensures that no frequencies above the original Nyquist frequency (100 Hz for the 200 Hz stream) are generated. It avoids the high-frequency spectral noise introduced by piecewise polynomials, aligning perfectly with the assumptions of the downstream STFT feature extractor.

### B. Lanczos Resampling (Windowed Sinc)
While sinc interpolation is theoretically perfect, it requires infinite support. Lanczos interpolation resolves this by windowing the sinc function with a second, wider sinc function:
\[L(t) = \text{sinc}(t) \cdot \text{sinc}\left(\frac{t}{a}\right) \quad \text{for} \quad -a < t < a\]
- **Downstream Benefit**: By limiting the spatial support to a local window (typically \(a=2\) or \(a=3\)), it minimizes the ringing artifacts (Gibbs phenomenon) that occur at sharp transitions in a pure sinc filter, while keeping high-frequency noise leakage to a minimum.

### C. Fourier (FFT-based) Spectral Interpolation
This method transforms the signal to the frequency domain using the Discrete Fourier Transform (DFT), pads the spectrum with zeros (high frequencies), and transforms it back via the Inverse DFT (IDFT).
- **Downstream Benefit**: This is the exact frequency-domain equivalent of sinc interpolation for periodic signals. It keeps the original spectral coefficients intact while perfectly interpolating new time-domain points on the upscaled grid, eliminating any polynomial fitting distortions.

---

## 3. Comparative Summary

| Interpolation Method | Optimization Objective | Main Drawback for SpeechBrain | Downstream Impact |
| :--- | :--- | :--- | :--- |
| **Spline (Cubic/Akima)** | Piecewise geometric smoothness (derivatives) | Polynomial boundary kinks, spectral leakage | Spurious high-frequency spectrogram noise, covariate shift |
| **Sinc (Whittaker-Shannon)** | Perfect band-limited reconstruction | Infinite temporal support (requires truncation) | Eliminates out-of-band noise; aligns with STFT assumptions |
| **Lanczos** | Windowed band-limited reconstruction | Moderate attenuation near cutoff | Balanced localized reconstruction; low spectral leakage |
| **Fourier (FFT)** | Spectral zero-padding | Assumes signal periodicity | Zero-distortion of low-frequency spectrum; ideal for cyclic components |

---

## 4. Experimental Results on Test Split (3,070 Sentences)

Below is the comparative evaluation of the three baseline models and the pipelines built on top of the new DSP-aligned interpolation variants:

| Model / Pipeline ID | Interpolation Method | Pre-Filter | Post-Filter | Signal MSE | Downstream Teacher WER / CER / SER (%) | Est. Student WER / CER / SER (%) | Status / Comparison |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **[BASELINE 1] Clean Limit** | *None* | *None* | *None* | *N/A* | 3.42% / 1.92% / 10.03% | 3.42% / 1.92% / 10.03% | Theoretical Upper Bound (No Cap) |
| **[BASELINE 2] STAG Original** | Cubic Spline | None | None | 1.033503 | 59.58% / 38.20% / 99.25% | 13.02% / 7.30% / 42.83% | Paper Reference Baseline |
| **[BASELINE 3] STAG + Post-Filter** | Cubic Spline | None | Butterworth (80Hz) | 0.535705 | 62.10% / 40.09% / 99.71% | 8.40% / 4.71% / 27.03% | Post-Filter Control Baseline |
| **Exp_V2 (Lanczos)** | Lanczos | None | None | 1.042174 | 60.10% / 38.55% / 99.30% | 13.10% / 7.35% / 43.11% | Regressed compared to Baselines 2 & 3 |
| **Exp_V5 (Sinc)** | Sinc | None | Butterworth (80Hz) | 0.536286 | 62.15% / 40.12% / 99.74% | 8.40% / 4.71% / 27.05% | Equalizes with Baseline 3; Beats Baseline 2 |

---

## 5. Implementation in `projects/interpolation_experiments/alternative_interpolations.py`
We have implemented these alternative interpolation functions:
- [alternative_interpolations.py](file:///c:/Users/jyoti/OneDrive/Desktop/STAG%20Implementation%20with%20StealthyIMU%20VUI/projects/interpolation_experiments/alternative_interpolations.py)

You can run experiments using `sinc` or `lanczos` for pre-prediction alignment to evaluate their impact on the down-stream SpeechBrain model's WER/CER metrics.
