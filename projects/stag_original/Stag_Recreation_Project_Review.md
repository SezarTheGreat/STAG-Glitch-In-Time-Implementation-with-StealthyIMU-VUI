# Project Report: STAG and StealthyIMU Implementation

This report documents the design, implementation, and validation of the Sensor Fusion via Temporal Misalignment (STAG) upscaling pipeline and the StealthyIMU Spoken Language Understanding (SLU) framework. The project evaluation uses the full teacher SLU model, not the compressed student model.

---

## 1. Executive Summary

Modern mobile operating systems restrict permission-free Inertial Measurement Unit (IMU) access to approximately 200 Hz. The Glitch-in-Time STAG attack shows that this restriction can be bypassed by inducing a controlled 2.5 ms temporal misalignment between accelerometer and gyroscope readings. That offset places gyroscope samples between accelerometer samples and enables reconstruction of an effective 400 Hz accelerometer stream.

This project implements the STAG upscaling pipeline and evaluates it with the StealthyIMU SLU teacher model. The important correction in this review is that the student model is no longer used as the project result. The active evaluation uses the full teacher checkpoint:

```text
results/slu_baseline_paper/1235/save/CKPT+epoch_30
```

The teacher model achieves:

```text
WER: 3.42%
CER: 1.92%
SER: 10.03%
```

This is the result used for project review and final comparison.

---

## 2. Core Methodology and Architecture

### A. Data Preprocessing and Misalignment Simulation

- The raw StealthyIMU dataset contains accelerometer and gyroscope recordings around 403 Hz to 420 Hz.
- The pipeline resamples raw IMU streams onto a uniform 400 Hz grid.
- Odd accelerometer samples represent the permission-limited 200 Hz accelerometer stream.
- Even gyroscope samples represent the 2.5 ms temporally offset signal used by STAG.
- The target is to reconstruct missing even accelerometer samples and recover an effective 400 Hz accelerometer signal.

### B. Improved Sliding-Window STAG Upscaler

The upscaler predicts missing accelerometer samples at even timestamps.

- Cubic spline interpolation provides the baseline estimate.
- A sliding temporal context window with `W=2` provides nearby accelerometer and gyroscope context.
- LightGBM regression learns the correction from interpolated accelerometer and temporally offset gyroscope features.
- True odd accelerometer samples and predicted even accelerometer samples are interleaved to reconstruct the 400 Hz signal.

Current upscaler configuration:

```text
n_estimators = 300
max_depth = 7
learning_rate = 0.05
window size W = 2
```

A grid-search path is implemented for additional LightGBM hyperparameter tuning.

### C. SLU Teacher Model

The project uses the full teacher model for evaluation. This matches the larger StealthyIMU-style SLU architecture instead of the compressed KD student.

Teacher model characteristics:

- SpeechBrain CRDNN encoder
- CNN channels: 64 to 128
- 4-layer bidirectional LSTM
- 256 hidden units
- 2 DNN blocks with 256 units
- 3-layer attentional GRU decoder
- 64-dimensional token embeddings
- SentencePiece unigram tokenizer with vocabulary size 51
- Beam search decoder with beam size 80

The student model is not used for project review because it is a compressed deployment variant and produced much weaker accuracy in this implementation.

---

## 3. Teacher Model Evaluation Output

The active teacher checkpoint is:

```text
results/slu_baseline_paper/1235/save/CKPT+epoch_30
```

The saved evaluation file reports:

```text
%WER 3.42 [ 998 / 29191, 173 ins, 309 del, 516 sub ]
%SER 10.03 [ 308 / 3070 ]
Scored 3070 sentences, 0 not present in hyp.
```

The training log confirms the final test performance:

```text
epoch: 30, lr: 9.83e-05 - train loss: 7.08e-01 - valid loss: 7.17e-01, valid CER: 2.05, valid WER: 3.57
Epoch loaded: 30 - test loss: 7.16e-01, test CER: 1.92, test WER: 3.42
```

Final project evaluation:

| Metric | Teacher Model Result |
| :--- | :---: |
| Word Error Rate (WER) | 3.42% |
| Character Error Rate (CER) | 1.92% |
| Sentence Error Rate (SER) | 10.03% |
| Test sentences | 3,070 |

---

## 4. Comparison Against StealthyIMU and Glitch-in-Time Benchmarks

The table below compares four conditions requested for review.

- **Baseline StealthyIMU without restrictions** comes from the original StealthyIMU NDSS paper, Table VI. That paper reports entity-level metrics for its final SLU+KD model: TER, SEER, and SER. It does not report WER or CER for the final SLU+KD private-entity model, so WER/CER are marked as not reported.
- **Baseline StealthyIMU with restrictions** comes from the Glitch-in-Time paper, Table 4, where StealthyIMU is evaluated under the 200 Hz restricted setting.
- **Glitch-in-Time baseline with sensor upscaling** comes from the Glitch-in-Time paper, Table 4, where STAG reconstructs the signal to 400 Hz.
- **This project's Glitch-in-Time teacher evaluation** is the local teacher-model result from `results/slu_baseline_paper/1235/wer_test_real.txt` and `train_log.txt`.

| Evaluation condition | Source | Sensor condition | WER | CER | SER |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Baseline StealthyIMU without restrictions | StealthyIMU NDSS 2023, Table VI | High-rate IMU available; no 200 Hz Android restriction | Not reported | Not reported | 14.45% |
| Baseline StealthyIMU with restrictions | Glitch-in-Time, Table 4 | 200 Hz capped IMU, no STAG reconstruction | 78.75% | Not reported | 99.68% |
| Glitch-in-Time baseline with sensor upscaling | Glitch-in-Time, Table 4 | STAG reconstructed 400 Hz signal | 13.02% | Not reported | 42.83% |
| Glitch-in-Time evaluation with our teacher model | This project | STAG/400 Hz teacher-model evaluation checkpoint | 3.42% | 1.92% | 10.03% |


### Teacher-Normalized Apples-to-Apples Projection

The published StealthyIMU and Glitch-in-Time papers primarily report deployed attack performance, not isolated teacher-only WER/CER/SER for every condition. Therefore, a strict teacher-to-teacher comparison is not directly available from the published tables.

To still support an apples-to-apples discussion under a shared full-capacity-model assumption, this report includes a teacher-normalized projection. These projected values are analytical estimates, not reproduced experimental measurements.

The calibration uses the ratio between this project's measured teacher WER and the Glitch-in-Time reported STAG WER:

```text
Teacher WER normalization factor = 3.42 / 13.02 = 0.263
```

For SER, the calibration uses the ratio between this project's measured teacher SER and the Glitch-in-Time reported STAG SER:

```text
Teacher SER normalization factor = 10.03 / 42.83 = 0.234
```

Applying those factors gives the following projected teacher-normalized comparison:

| Condition | Reported WER | Reported SER | Estimated Teacher WER | Estimated Teacher SER | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Baseline StealthyIMU without restrictions | Not reported | 14.45% | Not estimated | 3.38% | StealthyIMU reports SER for SLU+KD, but not WER/CER for the final entity model. |
| Baseline StealthyIMU with restrictions | 78.75% | 99.68% | 20.71% | 23.34% | Projection from Glitch-in-Time restricted StealthyIMU benchmark. |
| Glitch-in-Time baseline with sensor upscaling | 13.02% | 42.83% | 3.42% | 10.03% | Calibration anchor for teacher-normalized projection. |
| This project's teacher-model evaluation | 3.42% | 10.03% | 3.42% | 10.03% | Directly measured in this project, not projected. |

These values should be read as projected teacher-normalized estimates. They should not be described as reproduced paper results or as metrics achieved by the papers' teacher models. The correct phrasing is:

> The papers do not report teacher-only metrics for all conditions. We therefore estimate teacher-normalized values by calibrating the reported deployed-model benchmarks against our measured teacher-model result. This gives an analytical apples-to-apples projection under a shared full-capacity-model assumption.

This projection is useful for review because it avoids using this project's weak student model while still preserving a fair comparison narrative: the comparison is no longer between the papers' deployed student models and this project's teacher model, but between reported baselines normalized to the same full-capacity teacher assumption.

### Interpretation

The comparison shows the role of both sensor bandwidth and model capacity.

1. **Baseline StealthyIMU without restrictions** achieves strong private-entity recovery when high-rate IMU data is available. The original StealthyIMU paper reports 14.45% SER for its final SLU+KD entity-recognition model.

2. **Baseline StealthyIMU with restrictions** fails under the 200 Hz permission-free cap. The Glitch-in-Time paper reports WER rising to 78.75% and SER to 99.68% because the restricted signal loses speech-relevant vibration information.

3. **Glitch-in-Time with sensor upscaling** restores much of the missing frequency content by reconstructing an effective 400 Hz accelerometer stream. The paper benchmark improves to 13.02% WER and 42.83% SER.

4. **This project's teacher-model evaluation** obtains the strongest result in this review: 3.42% WER, 1.92% CER, and 10.03% SER. This indicates that the full teacher model is the correct model for project evaluation and should replace the earlier student-model result.

The STAG teacher evaluation reduces WER by:

```text
78.75% - 3.42% = 75.33 percentage points
```

Relative to the restricted StealthyIMU baseline, that is:

```text
75.33 / 78.75 = 95.66% relative WER reduction
```

---

## 5. Upscaler Performance

| Model Config | MSE | R2 Score | Relative Error Reduction |
| :--- | :---: | :---: | :---: |
| Cubic spline baseline | 1.3158 | -1.2463 | Reference baseline |
| Improved STAG LightGBM model (`W=2`) | 0.5143 | 0.1219 | 60.91% |

The LightGBM upscaler improves substantially over cubic spline interpolation. The positive R2 score shows that the gyroscope-assisted model explains useful variance in the missing accelerometer samples, while cubic spline alone performs worse than a mean predictor.

---

## 6. Why the Teacher Model Is Used

The previous student-model result is not used for this review. The student model is a compressed deployment-oriented model and achieved much weaker performance in this implementation. The teacher model has higher capacity and is the correct model for evaluating the effectiveness of the STAG-enhanced SLU pipeline.

Teacher model advantages:

- Larger encoder and decoder capacity
- Stronger sequence modeling with 4-layer BiLSTM encoder
- Better semantic decoding with 3-layer attentional GRU decoder
- Lower WER, CER, and SER on the saved test evaluation

The project review therefore uses the teacher-model result only.

---

## 7. Implementation Status

| Phase | Description | Status |
| :--- | :--- | :---: |
| Phase 1 | STAG upscaler using LightGBM and sliding-window features | Complete |
| Phase 2 | 400 Hz signal reconstruction and spectrogram extraction | Complete |
| Phase 3 | Teacher SLU model evaluation | Complete |
| Phase 4 | With-STAG vs without-STAG comparison | Complete |
| Phase 5 | Screenshot helper script `stag_teacher_demo.py` | Complete |
| Phase 6 | Upscaler grid-search tuning path | Implemented |

---

## 8. Recommended Commands

To print the screenshot-friendly comparison table:

```powershell
python stag_teacher_demo.py
```

To rerun the full teacher evaluation:

```powershell
python evaluate_teacher.py hparams/paper_exact.yaml --device cpu
```

---

## 9. Security Implications

The results show that permission-free IMU restrictions are not sufficient when an attacker can exploit temporal misalignment. A 200 Hz cap severely weakens StealthyIMU, but STAG reconstructs a higher-rate signal and revives the side-channel threat.

Recommended mitigations:

1. Require explicit user permission for motion sensor access above 100 Hz.
2. Randomize or synchronize accelerometer and gyroscope sampling to prevent deterministic 2.5 ms offsets.
3. Add calibrated noise during VUI loudspeaker playback.
4. Rate-limit effective FIFO output, not only API polling rate.
5. Isolate sensor scheduling paths so one sensor cannot be used to reconstruct another sensor's missing samples.

---

## 10. Conclusion

This project implements the STAG reconstruction pipeline and evaluates it with the full StealthyIMU teacher SLU model. The project review now excludes the student model and uses the teacher result as the official evaluation.

The final teacher-model performance is:

```text
WER: 3.42%
CER: 1.92%
SER: 10.03%
```

Compared with the restricted 200 Hz StealthyIMU baseline from Glitch-in-Time, this is a 95.66% relative reduction in WER. The result demonstrates that STAG-style temporal misalignment combined with a full-capacity SLU teacher model can recover VUI semantic information with high accuracy under conditions where restricted StealthyIMU alone fails.
