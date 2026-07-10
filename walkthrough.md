# Project Walkthrough: Advanced Signal Denoising and Upscaling

This document summarizes the changes made to the STAG upscaler and StealthyIMU VUI evaluation pipeline, the testing procedures, and the final validation results.

---

## 1. Changes Implemented

### Upscaling Algorithms (`projects/interpolation_experiments/pipeline_variants.py`)
*   **Variant 1 (Higher-Order B-Splines)**: Replaced standard cubic splines with a 5th-order B-Spline pre-alignment interpolation.
*   **Variant 2 (Pre-Interpolation Kalman)**: Integrated a kinematic state-space Kalman Filter and Rauch-Tung-Striebel (RTS) smoother to clean 200 Hz raw acc/gyro input streams.
*   **Variant 3 (Post-Correction Filter)**: Appended an 80 Hz Low-Pass Butterworth Filter to the final, interleaved 400 Hz output of the upscaler to remove LightGBM piecewise step artifacts.
*   **Variant 4 (Combined Pre & Post)**: Integrated both front-end Kalman RTS smoothing and back-end Butterworth filtering to create a noise-resilient upscaling pipeline.

### Pipeline and Evaluation (`projects/interpolation_experiments/evaluate_variants.py`)
*   Fixed the temporal alignment bug in the audio pipeline by parsing the exact duration of the spoken command directly from the `.wav` file (supporting IEEE Float formats via `scipy.io.wavfile`).
*   Modified evaluation logic to run all five variants sequentially.
*   Implemented dynamic linear projection equations to estimate ASR Student Model downstream performance.

---

## 2. Validation & Testing Results

*   Verified correct functionality on a 10-sentence subset to check formatting and float PCM compatibility.
*   Successfully ran the complete evaluation on the full test split of **3,070 sentences** for all configurations.
*   Auto-generated the review tables and analysis reports in `Interpolation_methods_review.md` and `combined_filter_pipeline.md`.
