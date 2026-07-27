# Day 13 Repository Findings

## Overview
This document captures repository structure discovery, codebase inspection findings, pre-existing utilities relevant to Day 13, and dependency analysis.

## 1. Folder Hierarchy & Experiment Structure
The workspace is structured chronologically by experiment days:
- `common/`: Contains shared logic, including `interpolation/interpolation.py` (`cubic_spline_interpolate`, `linear_interpolate`) and `data/` (`StealthyIMU_dataset`).
- `models/`: Stores model weights and checkpoints (`inertiear/`, `stag/`, `stealthy_imu/`).
- `day_04_05_stag_recreation/`: Core STAG baseline implementations, LightGBM upscaler, Seq2Seq SLU models (with BiGRU), dataset processing, and evaluation metrics.
- `day_07_08_alternate_models_and_interpolation/`: Variant evaluations, including CNN/RNN models and LightGBM configurations.
- `day_09_10_filter_pipelines_and_optimizations/`: Filtering algorithms, including Wiener filtering (`algorithms.py`), GRU correctors, and signal processing pipelines.
- `day_11_12_boosting_and_peaking/`: Signal boosting and adaptive Wiener filtering experiments.

## 2. Reusable Implementations Discovered

| Component | Found? | Location / Notes |
| :--- | :--- | :--- |
| **STAG LightGBM Upscaler** | Yes | `day_04_05/src/models/upscaler.py` |
| **InertiEAR Preprocessing** | Partial | Signal filtering exists in `day_09_10`, but exact InertiEAR pipeline requires assembly. |
| **StealthyIMU** | Yes | Dataset loading in `day_04_05/src/pipeline/dataset.py`. |
| **Interpolation** | Yes | `common/interpolation/interpolation.py` (Cubic Spline, Linear). |
| **Wiener Filtering** | Yes | `day_09_10_filter_pipelines_and_optimizations/algorithms.py` and `day_11_12_boosting_and_peaking/method2_wiener.py`. |
| **Seq2Seq & GRU** | Yes | `day_04_05_stag_recreation/src/models/slu_dnn.py` contains BiGRU and Seq2Seq with attention. |

## 3. Missing Implementations (To Be Built)
The following required components for the Day 13 Hybrid Pipeline do not currently exist and must be developed from scratch:
- **Otsu Thresholding** (for Automatic Speech Segmentation)
- **DenseNet** (for Branch B targeted vocabulary inference)
- **BLSTM** (for Branch A continuous speech decoder, replacing BiGRU)
- **Sensor Fusion (Accel/Gyro Alignment)** (Specific InertiEAR alignment logic)
- **Device Independent Normalization**
