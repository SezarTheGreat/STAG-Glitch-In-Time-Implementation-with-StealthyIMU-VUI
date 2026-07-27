# Day 13 Dataflow Diagram

## Tensor & Signal Dataflow

```mermaid
graph LR
    A["Raw Data (N, 6, T_raw)"] -->|Module 1| B["Denoised (N, 6, T_raw)"]
    B -->|Module 2| C["Segmented Mask (N, T_raw)"]
    B -->|Mask Apply| D["Speech Windows (K, 6, T_sub)"]
    D -->|Module 3| E["Normalized Feats (K, D_norm, T_sub)"]
    E -->|Module 4| F["Reconstructed (K, 1, 2*T_sub)"]
    
    F -->|Module 5A| G["Spectrogram (K, F, T_spec)"]
    G --> H["Text Transcriptions"]
    
    F -->|Module 5B| I["Spectrogram 244x244"]
    I --> J["Vocabulary Logits"]
```

## Data Types and Rates
- **Raw IMU**: 200 Hz (float32)
- **STAG Output**: 400 Hz (float32)
- **Spectrogram**: Log-Mel or STFT complex magnitudes (float32)
