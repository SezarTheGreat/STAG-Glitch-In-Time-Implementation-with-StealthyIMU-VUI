# Day 13 Dependency Graph

## System-Level Dependencies

```mermaid
graph TD
    A[StealthyIMU Dataset 200Hz] --> B(Module 1: Raw Data Capture & Intrinsic Denoising)
    B --> C(Module 2: Automatic Speech Segmentation)
    C --> D(Module 3: Device Independent Normalization)
    D --> E(Module 4: STAG Reconstruction Engine)
    
    E --> F{Module 5: Dual Branch Inference}
    
    F -->|Branch A: Unconstrained Speech| G[Seq2Seq Network]
    G --> G1(Embedding)
    G1 --> G2(CNN Encoder)
    G2 --> G3(BLSTM)
    G3 --> G4(Attention)
    G4 --> G5(GRU Decoder)
    
    F -->|Branch B: Targeted Vocabulary| H[DenseNet Network]
    H --> H1(244x244 Spectrogram Gen)
    H1 --> H2(DenseNet Classifier)
```

## Reusable Asset Dependencies
- **`common/interpolation.py`**: Required by Module 2 (Linear) and Module 4 (Cubic Spline).
- **`day_09_10_filter_pipelines/algorithms.py`**: Wiener filter logic reused in Module 1.
- **`day_04_05_stag_recreation/src/models/upscaler.py`**: LightGBM foundation utilized in Module 4.
