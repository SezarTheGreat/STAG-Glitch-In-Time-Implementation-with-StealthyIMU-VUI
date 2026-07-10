| Pipeline Stage | Original STAG Paper Dataset Framework | Our StealthyIMU Replication Framework |
| :--- | :--- | :--- |
| **Data Collection / Sampling** | Native, unthrottled hardware layer logging at true 400 Hz. | Static time-series IMU profiles derived from the public StealthyIMU repository. |
| **Temporal Glitch Mechanism** | Natural hardware-level 2.5 ms offset induced when integrating the magnetometer sensor. | Manually simulated 2.5 ms offset achieved by software-level timestamp shift / windowing. |
| **Model Training (Phase 1 & 2)** | Custom 400 Hz speech corpus split into Odd (Input Features) and Even (True Target Labels). | StealthyIMU matrices artificially downsampled or bifurcated to odd/even sequences to act as answers. |
| **Downstream Evaluation (Inference)** | End-to-end VUI sentence processing evaluated via Word Error Rate (WER) on spoken audio targets. | Signal reconstruction tracking (MSE / R² Fit) and classification against StealthyIMU command schemas. |

## Dataset Usage by Architecture Step

This section maps the phases from `STAG Model Architecture understanding.docx` to the dataset actually used in the original STAG paper framework and in this StealthyIMU-only replication.

| Document Step / Phase | What the Step Does | Dataset in Original STAG Paper | Dataset in Our Implementation |
| :--- | :--- | :--- | :--- |
| **Training Time - Step 1: The Glitch** | Establishes the 2.5 ms temporal gap where gyroscope samples fall between accelerometer samples. | The paper uses a hardware-level timing condition induced during collection, with the magnetometer integration creating naturally staggered accelerometer/gyroscope timing. | We do not have the authors' naturally misaligned collection. We synthesize the same 2.5 ms relationship by resampling StealthyIMU `.acc` and `.gyro` files onto a uniform 400 Hz grid, then offsetting the gyro/target side by one 400 Hz sample. |
| **Training Time - Step 2: Training Setup / Answer Key** | Builds supervised pairs by hiding half of the accelerometer stream and using the hidden half as labels. | Uses the authors' custom high-rate speech-vibration corpus. The 400 Hz accelerometer stream is split into visible 200 Hz odd accelerometer inputs and hidden 200 Hz even accelerometer labels. | Uses public StealthyIMU sensor recordings from the training split. The raw traces are normalized, median-filtered, resampled to 400 Hz, and bifurcated into `acc_odd`, `gyro_even`, and `acc_even_target`. |
| **Training Time - Step 3A: Cubic Spline Solver** | Creates a smooth baseline estimate of missing accelerometer points from visible accelerometer samples. | Runs on the visible 200 Hz accelerometer stream derived from the paper's 400 Hz corpus. | Runs on `acc_odd` from StealthyIMU after synthetic bifurcation. This provides `acc_interp` at the hidden even timestamps. |
| **Training Time - Step 3B: LightGBM Phase 1 / Intelligent Translation** | Learns to predict missing accelerometer samples using accelerometer context and staggered gyroscope hints. | Trains on paper-collected staggered sensor pairs, where the gyroscope timing reflects the physical offset. Labels are the withheld even accelerometer samples. | Trains on synthetic StealthyIMU pairs: input features are `acc_odd`, cubic-interpolated accelerometer estimates, and `gyro_even`; labels are `acc_even_target`. |
| **Training Time - Step 4: Final Fusion / Stitching** | Combines the smooth interpolation estimate with gyro-assisted predictions to recover the missing accelerometer stream. | The paper's final reconstruction is validated against the held-out even accelerometer labels from its own 400 Hz corpus. | Our implementation reconstructs by interleaving true `acc_odd` with predicted `acc_even`. Reconstruction quality is validated against the StealthyIMU-derived 400 Hz baseline using MSE and R2. |
| **Testing/Validation Time - Phase 1: Catching Real-Time Streams** | Captures restricted 200 Hz accelerometer and 200 Hz gyroscope streams during inference. | Uses live or collected phone streams where the sensor timing offset is naturally present because of the hardware-level glitch. No artificial split is required at inference time. | Uses validation/test StealthyIMU files, not live phone streams. The same synthetic bifurcation procedure is applied to create a testing-time equivalent of visible `acc_odd` and shifted `gyro_even`. |
| **Testing/Validation Time - Phase 2: Live Assembly Line** | Applies cubic spline interpolation and LightGBM prediction to the available 200 Hz streams. | Applies the trained STAG pipeline to naturally staggered 200 Hz accelerometer/gyroscope streams. | Applies the trained pipeline to synthetic StealthyIMU validation/test pairs. The target `acc_even_target` is retained only for scoring, not as an inference input. |
| **Testing/Validation Time - Phase 3: 400 Hz Reconstruction** | Produces the final 400 Hz accelerometer stream and evaluates downstream speech recovery. | Reconstructed 400 Hz accelerometer data is evaluated in the VUI/SLU task using WER against spoken targets. | Reconstructed 400 Hz accelerometer data is evaluated mainly with signal-level MSE/R2 and, where applicable, command-schema classification or local SLU-style evaluation using StealthyIMU labels. |

## Phase 1 and Phase 2 Dataset Summary

| Phase | Original STAG Paper | Our StealthyIMU-Only Replication |
| :--- | :--- | :--- |
| **Phase 1: Training Time** | Proprietary/custom 400 Hz speech-vibration dataset collected by the authors under the STAG hardware timing setup. This dataset provides both the visible input stream and the hidden even accelerometer answer key. | Public StealthyIMU training split. Since it is not the authors' STAG collection, we construct the answer key by resampling to 400 Hz and splitting into visible odd accelerometer samples plus hidden even accelerometer targets. |
| **Phase 2: Testing / Validation Time** | Naturally staggered 200 Hz accelerometer and gyroscope streams from the STAG attack setup, evaluated end-to-end on VUI speech recovery. | Public StealthyIMU validation/test split. We again synthesize the staggered timing relationship, reconstruct the missing accelerometer samples, and compare against the StealthyIMU-derived baseline trace. |

## Compromises Made in the Replication

1. **Dataset substitution:** The original paper's proprietary, naturally misaligned 400 Hz speech-vibration corpus is unavailable. We substitute public StealthyIMU recordings, so the replication tests the method under a dataset transfer setting rather than the exact paper data framework.

2. **Synthetic temporal glitch:** The paper relies on a hardware-level 2.5 ms offset. Our implementation creates this offset in software by resampling and bifurcating timestamps. This preserves the learning geometry of the attack but does not prove that the same hardware scheduling behavior was reproduced.

3. **Artificial answer key construction:** In the paper, the hidden even accelerometer labels come from native high-rate sensor collection. In our implementation, the labels are produced by treating the resampled StealthyIMU 400 Hz trace as the baseline and hiding every other accelerometer sample.

4. **Inference is replay-based, not live:** The paper's inference setting uses live restricted 200 Hz streams with natural staggering. Our validation/test setting replays static StealthyIMU files and applies the same synthetic staggered pairing used during training.

5. **Evaluation emphasis shifts:** The paper emphasizes downstream VUI sentence recovery through WER. Our replication must place more weight on reconstruction metrics such as MSE and R2 because the available public dataset and local pipeline do not fully reproduce the authors' end-to-end live attack environment.

6. **Gyroscope role is preserved but approximated:** In both frameworks, the gyroscope is an auxiliary hint used to infer missing accelerometer samples, not the training label. The compromise is that our `gyro_even` timing is created through controlled index shifting rather than measured from a naturally glitched sensor stack.
