# Day 13 Pipeline Flowchart

## End-to-End Data Processing Flow

```mermaid
flowchart TD
    subgraph M1 [Module 1: Capture & Denoise]
        Raw[Raw Accel/Gyro Data (200Hz)] --> WF[Wiener Filtering]
        WF --> SNR[Stationary Noise Removal]
        SNR --> DCR1[DC Bias Removal]
    end

    subgraph M2 [Module 2: Speech Segmentation]
        DCR1 --> TLI[Temp Linear Interpolation]
        TLI --> AGA[Accel-Gyro Alignment]
        AGA --> AM[Axis Multiplication]
        AM --> DCE[DC Bias Extraction]
        DCE --> OT[Otsu Thresholding]
        OT --> SBD[Speech Boundary Detection]
    end

    subgraph M3 [Module 3: Normalization]
        SBD --> DR[Dimension Reduction]
        DR --> FN[Feature Normalization]
        FN --> DIS[Device Independent Scaling]
    end

    subgraph M4 [Module 4: STAG Engine]
        DIS --> CSI[Cubic Spline Interpolation]
        CSI --> LGBM1[LightGBM Pred Stage 1]
        LGBM1 --> LGBM2[LightGBM Fusion Stage 2]
        LGBM2 --> Recon[400Hz Recon Signal]
    end

    subgraph M5 [Module 5: Dual Inference]
        Recon --> BranchA[Branch A: Seq2Seq]
        Recon --> BranchB[Branch B: DenseNet]
    end
```
