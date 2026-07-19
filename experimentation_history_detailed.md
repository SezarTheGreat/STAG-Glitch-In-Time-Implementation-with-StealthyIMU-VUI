# STAG & StealthyIMU Detailed Experimentation History

This report maps the entire experimental trajectory of the STAG upscaling and StealthyIMU Speech Language Understanding (SLU) project, structured chronologically from **Experiment 1 (Recreation)** through **Experiment 6 (Iterative Filtering)**.

---

## Experiment 1: Recreation & Synthetic Data Framework
* **Objective**: Replicate the STAG framework's hardware-level temporal misalignment glitch in a software-simulated environment using public datasets.
* **Methodology**: 
  * resampled the raw StealthyIMU dataset to a uniform $400\text{ Hz}$ grid.
  * Programmed a synthetic $2.5\text{ ms}$ temporal offset by software-level timestamp shifting and index bifurcation: odd accelerometer samples form the visible $200\text{ Hz}$ stream; even gyroscope samples form the staggered helper stream.
  * Implemented a sliding temporal context window ($W=2$) and trained a LightGBM regressor to predict the missing even accelerometer samples.
* **Results**:
  * **Cubic Spline Baseline MSE**: $1.31580$
  * **LightGBM Reconstruction MSE**: $0.51430$ (a **60.91%** error reduction over spline).
* **Status**: **PASS**. Proved the mathematical viability of the learning geometry under synthetic bifurcation.

---

## Experiment 2: Dataset Alignment & Efficiency (Simulation vs. Reality)
* **Objective**: Address how STAG utilized the collected dataset despite lacking native hardware-level conditions in StealthyIMU, and examine upscaling efficiency.
* **Methodology**:
  * The original STAG paper relies on a hardware-level magnetometer integration glitch that naturally staggers accelerometer/gyroscope readings by $2.5\text{ ms}$. 
  * Lacking unthrottled hardware logs, we resampled StealthyIMU to $400\text{ Hz}$ and offset the gyroscope and target accelerometer streams by one sample in software.
  * Checked model efficiency. The LightGBM and sliding context window are highly lightweight, making the attack computationally feasible to run on simple virtual machines (e.g., DigitalOcean droplets) or directly on edge devices without high CPU/memory overhead.
* **Status**: **PASS (Design & Simulation)**. Successfully adapted public data to replicate proprietary hardware conditions and established resource-efficiency limits.

---

## Experiment 3: Upscaling Acc & Gyro for Audio Capture
* **Objective**: Test if STAG successfully upscales restricted $200\text{ Hz}$ accelerometer and gyroscope streams to a reconstructed $400\text{ Hz}$ stream to recover audio-induced chassis vibrations.
* **Methodology**:
  * Validated the upscaler's ability to interleave true odd accelerometer samples with predicted even accelerometer samples.
  * Extracted frequency-domain spectrogram features from the reconstructed $400\text{ Hz}$ waveform and fed them to the Speech SLU model for sentence/command decoding.
* **Results**:
  * Reconstructed $400\text{ Hz}$ signals successfully restored the missing $100\text{ Hz} - 200\text{ Hz}$ frequency band.
  * Downstream Speech Teacher evaluation on the upscaled $400\text{ Hz}$ signal improved WER from $78.75\%$ (restricted $200\text{ Hz}$ cap without STAG) to **3.42%** (with STAG).
* **Status**: **PASS**. Proved that STAG successfully recovers VUI semantic data by upscaling restricted IMU streams to a high-rate $400\text{ Hz}$ waveform.

---

## Experiment 4: Alternative Models, KD, and Downscaling (Day 7)
* **Objectives**: 
  1. Train the student Knowledge Distillation (KD) model for paper-to-paper inferencing.
  2. Implement alternative upscaling architectures (Random Forest, CNN, RNN, ensembles) to improve accuracy.
  3. Evaluate alternative interpolation methods (Akima, Lanczos, Sinc) to replace Cubic Splines.
  4. Define the downscaling mechanism.
* **Methodology & Findings**:
  * **Downscaling Mechanism**: Simulated Android's operating system cap by keeping only the odd-indexed samples of a $400\text{ Hz}$ stream (discarding even indices), forcing the signal down to $200\text{ Hz}$.
  * **Alternative Interpolator Benchmarks**:
    
    | Interpolation Method | Signal MSE | Est. Student WER (%) | Status / Key Takeaway |
    | :--- | :---: | :---: | :--- |
    | **Cubic Spline (Baseline)** | 1.033503 | 13.02% | Baseline control |
    | **Akima Spline (Exp_V1)** | **0.534724** | **8.39%** | Best overall spline with post-filter |
    | **Whittaker-Shannon Sinc (Exp_V5)** | **0.536286** | **8.40%** | Mathematically optimal for band-limiting |
    | **Lanczos (Exp_V2)** | 1.042174 | 13.10% | Regressed (lacks post-filtering control) |
  
  * **Alternative Model Architecture & Ensemble Benchmarks (Signal-Level)**:
    
    | Model / Ensemble Strategy | Mean Squared Error (MSE) | R-squared ($R^2$) Fit | Description |
    | :--- | :---: | :---: | :--- |
    | **LightGBM** | 0.42293 | 0.46876 | Tabular gradient boosted decision trees |
    | **Random Forest** | 0.42960 | 0.46037 | Flat sliding temporal context window |
    | **RNN (GRU)** | 0.42450 | 0.46678 | Captures sequence-level transitions |
    | **CNN (1D)** | 0.40881 | 0.48649 | Extracted spatial-temporal features |
    | **Stacking Ensemble (Ridge)** | **0.40137** | **0.49584** | **Best blend (Ridge L2 Meta-Regressor)** |

  * **Student Knowledge Distillation**: The full Speech Teacher is highly sensitive to covariate shift. To achieve proper paper-to-paper inference (projected $13.02\%$ student WER), a compressed Student model must be trained using KD, letting it co-adapt to the specific noise and artifacts of the upscaling pipeline.
* **Status**: **PASS**. Explored alternatives and proved that Sinc/Akima interpolations and KD training are required to optimize upscaler-to-speech accuracy.

---

## Experiment 5: Stacking/Ensemble Paradox & Bifurcation (Day 8)
* **Objectives**:
  1. Determine why the Stacking Ensemble did not work on the downstream model.
  2. Map dataset usage: verify if StealthyIMU contains gyroscope data and how bifurcation was done.
  3. Test pre-interpolation and post-correction filter combinations.
* **Methodology & Findings**:
  * **StealthyIMU Gyro & Bifurcation**: StealthyIMU does contain gyroscope data. By resampling both sensors to the same $400\text{ Hz}$ timeline, we bifurcated the matrices: odd-indexed accelerometer samples were kept as visible input; even-indexed gyroscope samples were used as the staggered helper stream.
  * **Why the Stacking Ensemble Failed Downstream**: The Stacking Ensemble achieved the best physical reconstruction (MSE: $0.35952$, explaining $54.84\%$ of variance). However, it changed the distribution of reconstruction noise. Because the pre-trained Speech Teacher was co-adapted specifically to legacy LightGBM artifacts, the cleaner signal from the ensemble appeared as Out-of-Distribution (OOD) data, causing the WER to spike to $72.15\%$.
  
  * **Filtering Configuration Comparison (Pre vs. Post vs. Combined Cascades)**:
    
    | Filter Configuration | Signal MSE | Est. Student WER (%) | Est. Student SER (%) | Status / Key Insight |
    | :--- | :---: | :---: | :---: | :--- |
    | **No Filter (Raw Baseline)** | 1.033503 | 13.02% | 42.83% | Control baseline |
    | **Variant 2 (Pre-Kalman Only)** | 0.781483 | 10.68% | 34.83% | Removes sensor electrical white noise |
    | **Variant 3 (Post-Butterworth Only)** | **0.535705** | **8.40%** | **27.03%** | **Best Performance; smooths step noise** |
    | **Variant 4 (Combined Pre & Post)** | 0.674275 | 9.68% | 31.42% | Over-smoothing (loss of micro-oscillations) |

  * **The Denoising Cascade Paradox (Over-Smoothing)**: Combined Pre-Kalman and Post-Butterworth underperformed compared to Post-Butterworth alone. This occurs because the Kalman filter's kinematic equations treat high-frequency voice vibrations as transient noise and flattens them before interpolation, creating a lossy chain of operations.
* **Status**: **FAIL (Ensemble Downstream/Cascade Denoising) / PASS (Physical Reconstruction)**. Solved the "Denoising Cascade" and "Covariate Shift" paradoxes.

---

## Experiment 6: Iterative Filtering & Step Noise Suppression (Day 9)
* **Objective**: Evaluate if iterative filtering (using the output of one filter stage as input for the next) suppresses LightGBM step noise and optimizes memory/parameters.
* **Methodology**:
  * Passed the interleaved raw output through multiple cascade iterations of the Butterworth/Wiener filter.
  * Investigated parameter efficiency: this multi-pass approach removes training-induced step artifacts while reducing active parameters by nearly half (from 500 to about half).
* **Limitations**:
  * **The Retraining Bottleneck**: If the model is only trained for a few epochs, any changes in the front-end filter require retraining the entire cascade from scratch to prevent error propagation.
* **Status**: **PASS (Step Noise Reduction) with Limitations (Retraining Bottleneck)**. Demonstrated a trade-off between parameter savings and training flexibility.

---

## Experiment 7: Advanced Pre-Filters, Post-Filters, and Gating Thresholds
* **Objective**: Evaluate alternative pre-filters, post-filters, and threshold-based gating strategies on a representative subset of the StealthyIMU test set (300 files) to optimize signal reconstruction and downstream accuracy.
* **Methodology**:
  * Implemented and tested:
    * **Pre-Filter Savitzky-Golay (5, 2)**: Smooths high-frequency transitions without phase lag by fitting a local polynomial.
    * **Pre-Filter Bandpass [2, 95] Hz**: Isolates speech band on raw signals.
    * **Post-Filter Savitzky-Golay (7, 3)**.
    * **Post-Filter Chebyshev Type II (80Hz)**.
    * **Post-Filter Elliptic (80Hz)**.
    * **Post-Filter Bandpass [5, 80] Hz & [10, 80] Hz**.
    * **Post-Filter Noise Gates (Hard 0.05 std, Soft 0.1 std)**: Suppresses sensor drift in silent periods.
* **Results (300-file test subset)**:

| Configuration | Signal MSE | Est. Student WER (%) | Est. Student CER (%) | Est. Student SER (%) | Status / Key Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Cubic Spline + LGB)** | 1.051120 | 13.02% | 7.30% | 42.83% | Reference Baseline |
| **Control (Post Butterworth 80Hz)** | 0.548823 | 8.43% | 4.73% | 27.16% | **Improves** (+47.79% MSE reduction) |
| **Pre-Filter Savitzky-Golay (5, 2)** | **0.537072** | **8.33%** | **4.67%** | **26.79%** | **Best Pre-Filter + Post-Butterworth** |
| **Post-Filter Chebyshev Type II (80Hz)** | **0.537570** | **8.33%** | **4.67%** | **26.80%** | **Best Alternative Post-Filter** |
| **Post-Filter Bandpass [5, 80] Hz** | 0.589294 | 8.80% | 4.94% | 28.42% | Limits low-frequency gravity drift |

* **Key Insights**:
  1. **Savitzky-Golay Pre-Filtering**: The best-performing pipeline uses a **Savitzky-Golay pre-filter (window=5, polynomial=2) combined with a Butterworth post-filter (80Hz)**. This configuration achieves an MSE of **0.537072** and an estimated Student WER of **8.33%** (a 48.90% error reduction over baseline).
  2. **Why SavGol Works**: Fitting local polynomials to a sliding window smooths high-frequency noise while preserving peak amplitudes (acoustic impulses) and introducing **zero phase shift**, ensuring perfect time-domain alignment between streams before spline interpolation.
  3. **Role of Post-Filtering**: Post-filters are critical because the LightGBM upscaler's decision trees predict samples independently, creating high-frequency "step noise" at boundary points. Post-filtering smooths these ML artifacts and aligns the composite waveform to the downstream ASR model's frequency band.
* **Status**: **PASS**. Advanced filtering successfully pushed the performance ceiling below the initial post-filtering controls.

---

## Experiment 8: Feature Boosting & Vocal Resonance Amplification
* **Objective**: Evaluate filters that actively amplify/boost key speech components (e.g. fundamental frequency, speech energy envelope) on 300 StealthyIMU test files to see if targeted acoustic amplification can improve signal representations.
* **Methodology**:
  * Implemented and tested:
    * **High-Boost Filters (A=1.5 and A=2.0)**: Amplifies high-frequency details ($A \cdot \text{Signal} + (1-A) \cdot \text{Lowpass}$).
    * **Teager-Kaiser Energy Operator (TKEO) Boost (Gain=1.5 and Gain=2.5)**: Employs TKEO energy tracking ($\Psi[x(n)] = x(n)^2 - x(n-1)x(n+1)$) to dynamically calculate voice activity and apply non-linear gain multipliers.
    * **Parametric Peaking EQ Filters (80Hz, +6dB and 120Hz, +9dB)**: Uses second-order peaking biquad filters to boost vocal format frequencies/harmonics.
* **Results (300-file test subset)**:

| Configuration | Signal MSE | Est. Student WER (%) | Est. Student CER (%) | Est. Student SER (%) | Status / Key Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Cubic Spline + LGB)** | 1.051120 | 13.02% | 7.30% | 42.83% | Reference Baseline |
| **Control (Post Butterworth 80Hz)** | 0.548823 | 8.43% | 4.73% | 27.16% | Baseline control |
| **TKEO-Boosted (Gain=1.5)** | **0.546592** | **8.41%** | **4.72%** | **27.09%** | **PASS** (Slightly beats Control MSE) |
| **Peaking EQ (120Hz, +9dB)** | 0.549499 | 8.44% | 4.73% | 27.18% | Competitive harmonic amplification |
| **TKEO-Boosted (Gain=2.5)** | 0.561458 | 8.55% | 4.79% | 27.55% | Excess transient gain amplification |
| **High-Boost (A=1.5, 80Hz)** | 1.548422 | 17.56% | 9.85% | 58.35% | Regressed physically (high HF variance) |

* **Key Insights**:
  1. **TKEO Energy Tracking**: The Teager-Kaiser Energy Operator provides a non-linear tracking mechanism that estimates the instantaneous energy of the vibration signal. Using a low gain multiplier (1.5) dynamically boosts speech envelope components during active periods while ignoring silent static. It achieved the best physical MSE (**0.546592**) among the boosting filters.
  2. **High-Boost HF Variance**: High-boost configurations ($A=1.5, 2.0$) heavily amplify high-frequency components. This dramatically increases point-by-point signal variance relative to the smooth ground truth, leading to physical MSE regression. However, this high-frequency energy matches acoustic vocal resonances, which represents valuable classification cues.
  3. **Peaking EQ Resonances**: Parametric EQ boosts centered around fundamental speech harmonics ($80\text{Hz} - 120\text{Hz}$) isolate and highlight voice frequencies without introducing out-of-band high-frequency noise.
* **Status**: **PASS (TKEO & Peaking EQ) / FAIL (High-Boost Physical MSE)**. Demonstrated the trade-off between physical reconstruction metrics and targeted feature amplification.
