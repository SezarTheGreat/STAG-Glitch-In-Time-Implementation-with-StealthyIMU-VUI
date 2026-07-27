# Day 13 Engineering Journal

## Purpose
This engineering journal records repository discoveries, implementation decisions, architectural trade-offs, assumptions, debugging history, and validation outcomes specific to **Day 13 Experiment**.

---

## Record Format
Every journal entry follows this structure:

```markdown
### [YYYY-MM-DD] Entry Title
- **Repository Discoveries**: 
- **Implementation Decisions**: 
- **Architectural Trade-offs**: 
- **Assumptions**: 
- **Debugging History**: 
- **Validation Outcomes**: 
```

---

### [2026-07-27] Initialization & Scope Framing
- **Repository Discoveries**: Repository inspection deferred per initialization rules.
- **Implementation Decisions**: Established clean directory structure under `Day_13_Experiment`.
- **Architectural Trade-offs**: Isolated Day 13 context from prior days to ensure experimental clarity and zero legacy coupling.
- **Assumptions**: Day 13 will integrate InertiEAR preprocessing, STAG reconstruction backend, StealthyIMU VUI interface, and a dual inference stream.
- **Debugging History**: N/A (Initialization phase).
- **Validation Outcomes**: Workspace initialization complete.

### [2026-07-27] Phase 1 & 2: Repository Discovery and Architecture Design
- **Repository Discoveries**: Discovered reusable modules in `common/` (Interpolation, datasets), `day_04_05` (LightGBM, Seq2Seq SLU), and `day_09_10` (Wiener filtering, BiGRU). Missing implementations identified: Otsu Thresholding, DenseNet, BLSTM, Sensor Fusion logic.
- **Implementation Decisions**: Defined strict API contracts, dataflow, and a 5-module pipeline architecture to enforce modularity.
- **Architectural Trade-offs**: Relying on separate Branch A (Seq2Seq) and Branch B (DenseNet) increases training/inference complexity but properly serves dual research goals (unconstrained vs targeted speech).
- **Assumptions**: Existing dataset structures (`StealthyIMU_dataset`) and tensor shapes will be utilized in Module 1.
- **Debugging History**: N/A.
- **Validation Outcomes**: Architecture phase completed. Pending approval to start Module 1 implementation.

### [2026-07-27] Phase 3: Module 1 Capture & Denoising Implementation
- **Repository Discoveries**: N/A.
- **Implementation Decisions**: Implemented `module1_capture_denoise.py` with adaptive Wiener filtering and median filter for stationary noise, alongside a configuration class `PreprocessingConfig`.
- **Architectural Trade-offs**: Chosen to use simple mean subtraction for DC bias removal to keep computational overhead low, mirroring typical embedded constraints.
- **Assumptions**: Raw signal tensor shape is strictly `(Channels, Time)`.
- **Debugging History**: Installed `uv` package manager to ensure a reproducible test environment on Windows.
- **Validation Outcomes**: Validated dimension preservation and numerical correctness of filters. Tests passed successfully.

### [2026-07-27] Phase 4: Module 2 Automatic Speech Segmentation Implementation
- **Repository Discoveries**: N/A.
- **Implementation Decisions**: Translated the *InertiEAR* methodology into `module2_segmentation.py`. We compute an interaction energy envelope via axis multiplication, apply Otsu's method for dynamic thresholding, and smooth boundaries using `scipy.ndimage` binary morphology.
- **Architectural Trade-offs**: Morphological closing/opening operations on the binary mask (rather than mathematical convolution) were chosen for robustness and to prevent type casting issues when dealing with boolean arrays.
- **Assumptions**: Accelerometer and gyroscope data inputs are synchronized and span the same temporal grid out of Module 1. 
- **Debugging History**: Fixed an issue with `scipy.signal.convolve` mutating boolean masks into integers by switching to `scipy.ndimage.binary_closing` and `binary_opening`.
- **Validation Outcomes**: Validated that high-energy speech segments are appropriately masked while noise floors are rejected. Tests passed successfully.

### [2026-07-27] Phase 5: Module 3 Device Independent Normalization Implementation
- **Repository Discoveries**: N/A.
- **Implementation Decisions**: Implemented `module3_normalization.py` which computes Z-score (or robust scaling via IQR) calibration purely using the active speech segments indicated by Module 2's mask, and scales the entire continuous trace accordingly.
- **Architectural Trade-offs**: Allowed toggling between Standard Z-score and Robust scaling (Median/IQR) to counter potential long-tailed mechanical noise spikes that slip past earlier filters.
- **Assumptions**: Bypassing scaling when the speech mask is entirely false/empty ensures numerical stability and prevents divide-by-zero errors.
- **Debugging History**: Implemented zero-division safety using `np.where` inside the scaling formulas to inject `1.0` if standard deviation or IQR equals zero.
- **Validation Outcomes**: Unit tests confirm active speech segments align with mean ~0 and std ~1, while retaining temporal dimensionality matching. Tests passed successfully.

### [2026-07-27] Phase 6: Module 4 STAG Engine Implementation
- **Repository Discoveries**: Evaluated `day_04_05` upscaler implementation to model dataflow and feature shapes.
- **Implementation Decisions**: Engineered `module4_stag_engine.py` incorporating Cubic Spline Interpolation to scale normalized 200Hz signal slices up to 400Hz proxy-acoustic frequencies. Mocked out the LightGBM two-stage fusion interfaces so structural tensors could be validated.
- **Architectural Trade-offs**: Rather than fully duplicating `StagUpscaler` or pulling it completely, built a cleaner API boundary class `StagEngine` that wraps interpolation and encapsulates the feature construction logic for inference.
- **Assumptions**: The LightGBM tree fusion requires sliding temporal windows across the input channels, requiring `(6 channels) * (2W + 1)` extracted parameters per prediction point.
- **Debugging History**: None required during inference simulation mapping.
- **Validation Outcomes**: Confirmed that a 1000 sample 200Hz sequence strictly expands into a 2000 sample 400Hz acoustic proxy tensor `(1, 2T)`. Proved interpolation stability by validating that true target values persist precisely on even-sample boundaries.
