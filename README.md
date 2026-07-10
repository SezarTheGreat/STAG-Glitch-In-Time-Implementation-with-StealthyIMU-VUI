# STAG Implementation with StealthyIMU VUI

This repository serves as an **active machine learning experimentation environment** developed during an internship at the **Defense Research and Development Organisation (DRDO)**. The project focuses on researching, implementing, and optimizing sensor upscaling and denoising techniques to evaluate side-channel vulnerabilities in Spoken Language Understanding (SLU) and Voice User Interface (VUI) systems.

---

## 1. Project Context & Objectives

Modern mobile operating systems restrict permission-free Inertial Measurement Unit (IMU) access to approximately 200 Hz. The **Sensor Fusion via Temporal Misalignment (STAG)** attack framework demonstrates that this software restriction can be bypassed by exploiting a hardware-level 2.5 ms temporal misalignment between accelerometer and gyroscope readings. This staggered sampling allows for the reconstruction of an effective 400 Hz accelerometer stream from permission-free 200 Hz streams, exposing the device to motion-sensor-based eavesdropping.

### DRDO Internship Goals:
*   **Replicate** the STAG hardware-level glitch in a software-simulated environment using public datasets.
*   **Evaluate** the vulnerability transferability on the **StealthyIMU** dataset.
*   **Develop and Test** advanced machine learning reconstruction models (e.g., Stacking Ensembles).
*   **Explore and Analyze** signal denoising pipelines (Kalman RTS Smoothers, Butterworth Filters) to enhance reconstruction quality and understand downstream model co-adaptation.

---

## 2. Codebase Architecture

The project is structured logically, splitting core utilities, model definitions, and separate research pipelines:

```
├── .gitignore                                 # Git exclusions (datasets, local weights, caches)
├── README.md                                  # Comprehensive project documentation
├── walkthrough.md                             # Summary of changes and validation procedures
├── Dataset_Mismatches_Review.md               # Analysis of paper vs StealthyIMU differences
├── Interpolation_methods_review.md            # Benchmark report for interpolation variants
├── combined_filter_pipeline.md                # Flow diagrams and analysis for filtering variants
├── kalman_filter_pipeline.md                  # Documentation on state-space Kalman filters
├── post_filter_pipeline.md                    # Documentation on Butterworth post-correction
│
├── common/
│   └── interpolation/
│       └── interpolation.py                   # Shared signal interpolation helper functions
│
└── projects/
    ├── stag_original/                         # Replication of the original STAG paper setup
    │   ├── Stag_Recreation_Project_Review.md  # Original replication review
    │   ├── evaluate_teacher.py                # Raw teacher model evaluation script
    │   ├── run_phase2_kd.py                   # Phase 2 Knowledge Distillation script
    │   └── src/                               # Main source folder for original STAG models
    │
    ├── paper_based_reconstruction_with_other_models/ # Alternative upscaling architectures
    │   ├── paper cited model upscaling.md     # Reference notes on cited models
    │   ├── models.py                          # CNN, GRU, and MLP models
    │   └── evaluate.py                        # Evaluation harness for alternate models
    │
    ├── interpolation_experiments/             # Experiments with signal processing variants
    │   ├── pipeline_variants.py               # Implementation of B-spline, Kalman, and Butterworth
    │   ├── evaluate_variants.py               # Harness running all variants on 3,070 test set
    │   └── wer_*.txt                          # Log files containing raw WER test results
    │
    └── combined_reconstruction/               # Fused modeling combining ml and filters
        ├── Signal_Reconstruction_.md          # Stacking ensemble report
        ├── stacking.py                        # Stacking ensemble trainer/predictor
        └── evaluate_slu.py                    # Evaluator comparing baseline and stacker on SLU
```

---

## 3. Commit Classification Scheme

To maintain clear and structured development histories in this research environment, commits are categorized into distinct classes:

*   **`feat(...)`**: Introduces new functional features, model architectures, or experimentation pipelines (e.g., `feat(projects)`, `feat(common)`).
*   **`docs`**: Additions or updates to documentation, reports, mathematical writeups, or markdown summaries (e.g., `docs: add STAG reconstruction...`).
*   **`chore`**: Maintenance tasks, project configurations, or repository adjustments (e.g., `chore: update .gitignore`).
*   **`Cleanup / Refactor`**: Internal code restructuring, removal of deprecated testing scripts, or model-weight housekeeping.

---

## 4. ML Experimentation Log

### Experiment 1: Synthetic Glitch Framework & LightGBM Baseline
*   **Timestamp**: `2026-06-27 23:46:50` to `2026-06-28 00:52:52`
*   **Thinking**: Set up the initial simulation workspace. Since raw hardware unthrottled logging at 400 Hz was proprietary to the paper, the public StealthyIMU dataset was resampled to a uniform 400 Hz grid. The 2.5 ms temporal offset was simulated by software-level timestamp shifting and index bifurcation (odd accelerometer indexes representing the visible 200 Hz stream; even gyroscope indexes representing the staggered sensor helper stream).
*   **Method**: Implemented a Cubic Spline baseline upscaler and trained a downscaled LightGBM regressor (`W=2` sliding context window) using statistical feature extraction (mean, variance) and raw gyroscope values.
*   **Results**:
    *   Cubic Spline MSE: `1.31580`
    *   LightGBM Reconstruction MSE: `0.51430` (a **60.91%** error reduction over spline).
*   **Learnings**: Tabular gradient boosting combined with spatial window features is effective at mapping the non-linear relationship between adjacent accelerometer samples and staggered gyroscopes.

### Experiment 2: Teacher Model Evaluation & The Covariate Shift Paradox
*   **Timestamp**: `2026-06-28 17:47:40` to `2026-07-02 10:46:21`
*   **Thinking**: Evaluate the reconstructed signals against the pre-trained Speech Teacher SLU model (`results/slu_baseline_paper/1235/save/CKPT+epoch_30`).
*   **Method**: Constructed the evaluation pipeline to feed the reconstructed 400 Hz accelerometer waveforms into the Teacher SLU model to check speech recovery metrics (WER/CER/SER).
*   **Results**:
    *   Ground-Truth Baseline: WER `3.42%`, SER `10.03%`
    *   Reconstructed Baseline (Cubic+LGBM): WER `59.58%`, SER `99.25%`
*   **Learnings**: Discovered a massive drop in Teacher accuracy despite low physical signal MSE. The Teacher model was trained on clean signals, creating a severe **covariate shift** when exposed to the upscaler's prediction artifacts. This established the necessity of training a **Student model** via Knowledge Distillation (KD) to co-adapt to upscaler biases, scaling downstream WER to a projected `13.02%`.

### Experiment 3: Stacking Ensemble Upscaler
*   **Timestamp**: `2026-07-10 11:16:46` to `2026-07-10 11:17:18`
*   **Thinking**: Can we improve physical reconstruction metrics (MSE/R²) by combining multiple diverse modeling paradigms (tabular gradient boosting, deep sequence modeling, and neural spectrogram representation encoding)?
*   **Method**: Implemented a Stacking Ensemble consisting of:
    1.  **LightGBM**: Statistical windows + Wavelet (DWT) features.
    2.  **1D CNN**: Temporal convolution over raw windows.
    3.  **GRU**: Sequential recurrent tracking.
    4.  **Deep Speech Representation Encoder + MLP**: High-level acoustic states pooled from the teacher model.
    5.  **Ridge Meta-Regressor**: Fits a linear model with L2 regularization to blend base predictor outputs.
*   **Results**:
    *   Stacking Ensemble MSE: `0.35952`
    *   R² Fit Score: `0.54840` (**72.68%** relative error reduction over Spline).
    *   Teacher Model Evaluation (100-sentence subset): Baseline LGBM WER `49.22%` vs. Stacking Ensemble WER `72.15%`.
*   **Learnings**: The Stacking Ensemble achieved superior physical modeling (explaining `54.84%` of the variance compared to LGBM's `12.19%`). However, it triggered an even worse covariate shift in the static Speech Teacher model, which was heavily co-adapted to legacy LightGBM artifacts. This verified that to utilize advanced upscalers, the downstream ASR student must be retrained on the new upscaler outputs.

### Experiment 4: Signal-Processing and Denoising Pipeline Variants
*   **Timestamp**: `2026-07-10 11:17:01`
*   **Thinking**: LightGBM decision trees create piecewise constant outputs that manifest as high-frequency "step noise" in spectrograms. We want to evaluate if front-end state-space denoising (Kalman) and back-end signal smoothing (Butterworth) improve downstream evaluations.
*   **Method**: Evaluated four pipeline configurations on the full 3,070 test sentences:
    *   **Variant 1**: 5th-order B-Splines instead of cubic splines.
    *   **Variant 2 (Pre-Kalman)**: Kinematic Kalman Filter + RTS Smoother on raw 200 Hz streams.
    *   **Variant 3 (Post-Butterworth)**: 80 Hz Low-Pass Butterworth Filter applied to the 400 Hz interleaved signal.
    *   **Variant 4 (Combined)**: Pre-Kalman + Post-Butterworth.
*   **Results & Projections**:
    | Configuration | Signal MSE | Est. Student WER (%) | Est. Student SER (%) |
    | :--- | :---: | :---: | :---: |
    | **Baseline (Cubic + LGBM)** | 1.033503 | 13.02% | 42.83% |
    | **Variant 2 (Pre-Kalman)** | 0.781483 | 10.68% | 34.83% |
    | **Variant 3 (Post-Butterworth)** | **0.535705** | **8.40%** | **27.03%** |
    | **Variant 4 (Combined)** | 0.674275 | 9.68% | 31.42% |

---

## 5. Theoretical Analyses & Insights

### The Covariate Shift & Co-Adaptation Paradox
A physically superior signal (lower MSE) does not guarantee better speech decoding on a static pre-trained model. Neural speech encoders trained on specific noisy upscalers treat the artifacts as features. Replacing the upscaler changes the noise distribution, pushing the inputs Out-of-Distribution (OOD) for the acoustic decoder. Fine-tuning the ASR student via Knowledge Distillation on the target upscaler is mandatory to unlock accuracy gains.

### The Denoising Cascade Paradox (Over-Smoothing)
Variant 3 (Butterworth Post-Filter alone) outperforms Variant 4 (Combined Kalman + Butterworth) both in MSE (`0.535` vs `0.674`) and projected WER (`8.40%` vs `9.68%`). 

Speech vibrations propagating through the phone chassis are characterized by tiny, high-frequency, low-amplitude micro-oscillations. 
1.  The **Kalman Filter** employs a Constant Velocity kinematic state-space model, which assumes macro-scale physical movements and treats these high-frequency acoustic micro-vibrations as transient white noise.
2.  Smoothing raw signals at the 200 Hz front-end **flattens** these voice features before interpolation or machine learning upscaling occur.
3.  Applying a second filter at the back-end (Butterworth) creates a cascade of lossy operations, leading to **over-smoothing** where both noise and valid voice details are completely lost. 
4.  Leaving the raw 200 Hz streams untouched allows the LightGBM model to predict samples based on raw acoustic features. The post-reconstruction Butterworth filter then removes the artificial "step noise" above 80 Hz while preserving the voice band untouched.
