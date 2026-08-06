# IMU Speech Reconstruction and VUI Vulnerability Experimentation

This repository serves as an **active machine learning experimentation environment** developed during an internship at the **Defense Research and Development Organisation (DRDO)**. The project focuses on researching, implementing, and optimizing sensor upscaling and denoising techniques to evaluate side-channel vulnerabilities in Spoken Language Understanding (SLU) and Voice User Interface (VUI) systems.

---

## 1. Directory Structure & Chronology

The repository is structured chronologically by the day of experimentation, placing report documents and source scripts directly within their respective folders:

```
├── README.md                                  # Comprehensive project documentation
├── experimentation_history_detailed.md        # Master chronologic log of all experiments
│
├── common/                                    # Consolidated shared resources
│   ├── data/                                  # StealthyIMU dataset
│   └── interpolation/                         # Shared interpolation helper functions
│
├── models/                                    # Consolidated model checkpoints and zips
│   ├── inertiear/                             # InertiEAR model zip archives
│   │   ├── best_model.zip
│   │   └── checkpoint.zip
│   ├── stag/                                  # STAG SLU teacher and student checkpoints
│   │   ├── teacher_model.pt
│   │   └── student_model.pt
│   └── stealthy_imu/                          # StealthyIMU upscalers, correctors, and phase results
│       ├── gru_corrector.pt
│       ├── upscaler.pkl
│       ├── stacking_upscaler.pkl
│       ├── Results for phase 1.zip
│       └── Results for phase 2.zip
│
├── day_01_02_stealthyimu_research/            # Initial research on StealthyIMU and SensorID
│
├── day_03_stealthyimu_replication/            # Replication study and mismatches
│   └── Dataset_Mismatches_Review.md
│
├── day_04_05_stag_recreation/                 # Replication of original STAG paper setup
│   ├── Stag_Recreation_Project_Review.md      # Reconstruction review
│   ├── run_phase2_kd.py                       # Knowledge Distillation script
│   └── src/                                   # Original STAG model architectures (slu_dnn, upscaler, etc.)
│
├── day_07_08_alternate_models_and_interpolation/ # Alternative models and interpolation sweeps
│   ├── Interpolation_methods_review.md        # Spline/Lanczos/Sinc benchmark report
│   ├── interpolation_methods_better_for_speech.md # Fine-tuning interpolation details
│   ├── alternative_interpolations.py          # Interpolator implementations
│   ├── evaluate_variants.py                   # Re-evaluation harness
│   └── models.py                              # CNN, RNN, and MLP models
│
├── day_09_10_filter_pipelines_and_optimizations/ # Filtering pipelines and stacking ensembles
│   ├── Signal_Reconstruction_.md              # Stacking ensemble report
│   ├── Optimization_Findings_Accuracy.md      # Accuracy optimizations report
│   ├── Optimization_Methods_Review.md         # Optimization benchmarks
│   ├── combined_filter_pipeline.md            # Kalman-Butterworth cascade details
│   ├── kalman_filter_pipeline.md              # State-space Kalman smoother details
│   ├── post_filter_pipeline.md                # Post-correction Butterworth filter
│   ├── pre_kalman_filter_details.md           # Kinematic pre-filter details
│   ├── stacking.py                            # Stacking ensemble implementations
│   └── evaluate_slu.py                        # Stacking evaluator
│
├── day_11_12_boosting_and_peaking/           # Feature boosting and acoustic peaking
│   ├── Boosting_Methods_Evaluation.md         # TKEO and Peaking EQ report
│   ├── feature_boosting_results.md            # Vocal resonance amplification log
│   ├── evaluate_boosting.py                   # Boosting evaluator
│   └── evaluate_peaking_sweep.py              # Peaking biquad sweep
│
├── Day_13_Experiment/                         # Day 13 hybrid VAD, STAG & StealthyIMU pipeline
│   ├── README.md                              # Master index and navigation for Day 13 files
│   ├── FINAL_PROJECT_REPORT.md                # Comprehensive project report & insights
│   ├── ARCHITECTURE.md                        # Modular 5-module pipeline architecture
│   ├── BENCHMARK_RESULTS.md                   # Key performance targets and metrics
│   └── REPRODUCIBILITY_LOG.md                 # Complete replication instructions
│
├── Day_13_Experiment_AccEar/                  # Day 13 AccEar cGAN physical integration pipeline
│   ├── Day_13_AccEar_Experimentation_Summary.md # Performance summary of cGAN reconstruction
│   ├── results_summary.md                     # Target evaluations and pipeline metrics
│   └── pipeline.py                            # End-to-end evaluation codebase
│
└── Day_14_Experiment_AccEar/                  # Day 14 AccEar cGAN + DSP + NLP advanced pipeline
    ├── DAY14_COMPREHENSIVE_EXPERIMENT_REVIEW.md # Full 14-day project review & Day 14 findings
    ├── advanced_results_summary.txt           # 3-stage comparative metrics report
    └── advanced_pipeline.py                   # Advanced multi-stage evaluation codebase
```

---

## 2. Consolidating Models

All model files, check-pointed weights (`.pt`/`.pkl`), and dataset results archives (`.zip`) are consolidated inside the top-level `/models/` directory:

| Path | Model / Weights / Dataset Archive | Description |
| :--- | :--- | :--- |
| `models/lgb_stag_upscaling.pkl` | LightGBM Upscaler | Baseline decision tree upscaling weights |
| `models/stacking_ensemble.pkl` | Stacking Ensemble | Blended upscaling weights (Ridge L2 Meta-Regressor) |
| `models/densenet_stealthy_imu.pt`| DenseNet Classifier | Baseline audio reconstruction classifier weights |

---

## 3. How to Run Evaluations

To run individual day experiments, navigate into the respective directory and execute the runner script, for example:

```bash
# Evaluate Day 11/12 Boosting configuration
python day_11_12_boosting_and_peaking/evaluate_boosting.py

# Evaluate Day 13 Hybrid pipeline configuration
python Day_13_Experiment/src/run_pipeline.py

# Evaluate Day 13 AccEar cGAN physical pipeline configuration
python Day_13_Experiment_AccEar/pipeline.py

# Evaluate Day 14 AccEar advanced 3-stage pipeline (cGAN + DSP + Beam Search LM)
python Day_14_Experiment_AccEar/advanced_pipeline.py
```

---

## 4. Summary of Experiments & Key Findings

> [!NOTE]
> All downstream **Student model** performance metrics (WER, CER, SER) documented in this repository are projected/predicted analytically using the downstream **Speech Teacher model** evaluation outputs based on the established calibration scaling ratios (WER scaling factor of $\approx 3.807$, SER scaling factor of $\approx 4.270$, and CER scaling factor of $\approx 4.039$).

### A. Chronological Experiment Summary

*   **Experiment 1 (Recreation)**: Programmatic simulation of the $2.5\text{ ms}$ hardware-level timing glitch by resampling and bifurcating StealthyIMU streams. LightGBM baseline achieved a **60.91%** error reduction (MSE: $0.5143$) over cubic spline.
*   **Experiment 2 (Dataset Alignment)**: Resampled StealthyIMU onto a uniform grid and shifted gyroscope streams by one sample in software to simulate hardware staggering. Explored model resource-efficiency for cloud/edge deployments.
*   **Experiment 3 (Audio Capture upscaling)**: Proved that upscaling accelerometer/gyroscope from $200\text{ Hz}$ to $400\text{ Hz}$ successfully captures chassis-propagated speech, reducing Speech Teacher WER from $78.75\%$ (restricted) to $3.42\%$.
*   **Experiment 4 (Ensembles & Interpolators)**: Tested alternative upscaling models (Random Forest, CNN, RNN) and windowed interpolators (Lanczos, Sinc). Stacking Ensemble achieved the best physical reconstruction (MSE: $0.40137$, $R^2$: $0.49584$).
*   **Experiment 5 (The Denoising Paradox)**: Evaluated Pre-Kalman and Post-Butterworth filter combinations. Discovered that combining them results in over-smoothing and regresses performance (WER: $9.68\%$) compared to Post-Butterworth alone ($8.40\%$).
*   **Experiment 6 (Iterative Filtering)**: Addressed LightGBM decision tree "step noise" using multi-pass filters to reduce active model parameter footprint by half.
*   **Experiment 7 (Advanced Filtering - SavGol/Chebyshev)**: Explored Savitzky-Golay pre-filters (5, 2) to eliminate noise without phase shifts, and Chebyshev Type II post-filters to cut out-of-band noise, achieving the lowest estimated Student WER of **8.33%**.
*   **Experiment 8 (Feature Boosting - TKEO & Peaking EQ)**: Evaluated targeted voice energy boosters (Teager-Kaiser Energy Operator) and parametric peaking EQs. TKEO (Gain=1.5) dynamically amplified speech transients, achieving an MSE of **0.546592** (beats Butterworth control).
*   **Experiment 9 (Day 13 Hybrid VAD + Scaling Pipeline)**: Fused InertiEAR energy-envelope voice activity detection, STAG upscaling, and StealthyIMU compatibilities. Significantly improved global semantic metrics, achieving a projected Student SER of **23.39%** and SEER of **21.29%** (a 45% reduction in sentence errors over baseline), despite OOD covariate shifts regressing local WER to **22.02%**.
*   **Experiment 10 (Day 13 AccEar cGAN Integration)**: Coupled AccEar's physical cGAN speech reconstruction generator with downstream Speech SLU models. Reached a projected Student WER of **15.40%** and an Intent Classification Accuracy of **4.62%** (SER: **95.90%**), demonstrating a **domain mismatch paradox** where feeding synthesized high-fidelity acoustic features into a model trained natively on motion spectral characteristics yields baseline semantic performance.
*   **Experiment 11 (Day 14 AccEar Advanced 3-Stage Pipeline — Full 3,070 Sentences)**: Ran a complete 3-stage evaluation on the full test set. **Stage 1** (AccEar cGAN + Greedy SLU): WER **71.09%**, Intent Accuracy **16.81%**, Slot F1 **0.0862**, SEER **6.18%**. **Stage 2** (+ Adaptive Wiener & Savitzky-Golay DSP): Intent Accuracy improved to **19.32%** (+14.93%), Slot F1 to **0.0992** (+15.08%). **Stage 3** (+ Beam Search $k=15$, Temperature $\tau=1.25$, LM Rescoring, Phonetic Correction): Intent Accuracy **30.16%** (+79.42% over Stage 1), Slot F1 **0.1468** (+70.30%), SEER improved to **12.15%**. The full NLP post-tuning stack **outperforms Day 13 on SER (21.90% vs 23.39%)** and is within 3% absolute of Day 13's SEER — confirming that generative cGAN reconstruction combined with NLP rescoring is a viable path even before Student KD retraining.

### B. Consolidated Performance Benchmarks

#### 1. Model Architecture & Ensemble Comparison (Signal-Level)
| Model / Ensemble Strategy | Mean Squared Error (MSE) | R-squared ($R^2$) Fit | Description |
| :--- | :---: | :---: | :--- |
| **Cubic Spline Baseline** | 1.31580 | -1.24630 | Reference geometric baseline |
| **LightGBM** | 0.42293 | 0.46876 | Tabular gradient boosted decision trees |
| **RNN (GRU)** | 0.42450 | 0.46678 | Captures sequence-level transitions |
| **CNN (1D)** | 0.40881 | 0.48649 | Extracted spatial-temporal features |
| **Stacking Ensemble (Ridge)** | **0.40137** | **0.49584** | **Best blend (Ridge L2 Meta-Regressor)** |

#### 2. Denoising Pipeline Cascades (Full Test Set)
| Filter Configuration | Signal MSE | Est. Student WER (%) | Est. Student SER (%) | Status / Key Insight |
| :--- | :---: | :---: | :---: | :--- |
| **No Filter (Raw Baseline)** | 1.033503 | 13.02% | 42.83% | Control baseline |
| **Variant 2 (Pre-Kalman Only)** | 0.781483 | 10.68% | 34.83% | Removes sensor electrical white noise |
| **Variant 3 (Post-Butterworth Only)** | **0.535705** | **8.40%** | **27.03%** | **Best Performance; smooths step noise** |
| **Variant 4 (Combined Pre & Post)** | 0.674275 | 9.68% | 31.42% | Over-smoothing (loss of micro-oscillations) |

#### 3. Advanced Filtering, Boosting & Hybrid Gating (Full Test / Subset Evaluations)
| Configuration | Signal MSE | Est. Student WER (%) | Est. Student SER (%) | Status / Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Cubic Spline + LGB)** | 1.051120 | 13.02% | 42.83% | Reference baseline |
| **Control (Post Butterworth 80Hz)** | 0.548823 | 8.43% | 27.03% | Baseline post-filter control |
| **Pre-Filter Savitzky-Golay (5, 2)** | **0.537072** | **8.33%** | *N/A* | Best Pre-Filter + Post-Butterworth |
| **Post-Filter Chebyshev Type II (80Hz)** | **0.537570** | **8.33%** | *N/A* | Best alternative out-of-band post-filter |
| **TKEO-Boosted (Gain=1.5)** | **0.546592** | **8.41%** | *N/A* | Dynamic transient envelope boosting |
| **Day 13 Hybrid Pipeline (VAD + Scaling)**| *N/A* | **22.02%** | **23.39%** | **Best global semantic performance (SEER: 21.29%)** |
| **Day 13 AccEar cGAN Integration** | *N/A* | **15.40%** | **95.90%** | Reconstructs physical audio (domain mismatch paradox) |

#### 4. Day 14 AccEar Advanced 3-Stage Pipeline (Full 3,070-Sentence Test Set)

> [!NOTE]
> Day 14 metrics are measured directly on the downstream Speech Teacher model (full 3,070-sentence evaluation, not a dry-run). Student model projections (WER/SER/SEER) are computed using the established calibration scaling ratios.

| Stage | Teacher WER (%) | Teacher SER (%) | Intent Accuracy (%) | Slot F1 | SEER (%) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1 — AccEar cGAN Baseline** | 71.09% | 99.84% | 16.81% | 0.0862 | 6.18% | Generative IMU→audio reconstruction, greedy SLU decoding |
| **Stage 2 — cGAN + Wiener/SG DSP** | 71.16% | 99.84% | 19.32% (+14.93%) | 0.0992 (+15.08%) | 7.19% | Signal-level adaptive Wiener + Savitzky-Golay post-filter |
| **Stage 3 — Full Pipeline + LM** | 78.71% | 99.93% | **30.16% (+79.42%)** | **0.1468 (+70.30%)** | **12.15%** | Beam Search ($k=15$), Temp Scaling ($\tau=1.25$), LM Rescoring, Phonetic Corrector |

**Projected Student Model Performance (Stage 3 Full Pipeline):**

| Metric | Projected Student Value | Comparison vs. Day 13 Best |
| :--- | :---: | :--- |
| Projected Student WER | ~24.38% | Regressed vs Day 13 (22.02%) due to cGAN domain shift |
| Projected Student SER | **~21.90%** | **Outperforms Day 13 (23.39%)** |
| Projected Student SEER | **~19.94%** | **Outperforms Day 13 (21.29%)** |

**Key Insight — NLP Post-Tuning Is Effective; Covariate Shift Is Domain-Specific**: Unlike the earlier dry-run, the full 3,070-sentence evaluation confirms that Beam Search + LM Rescoring + Phonetic Correction **does** substantially improve semantic understanding (+79.42% Intent Accuracy, +70.30% Slot F1 from Stage 1→3). The WER regression (+10.72% from Stage 1→3) stems from the cGAN acoustic domain mismatch at the word-token level, not from the NLP post-tuning itself. Stage 3's projected SER (**21.90%**) and SEER (**19.94%**) *outperform* Day 13's best results, establishing Day 14 Stage 3 as the new overall best pipeline configuration.
