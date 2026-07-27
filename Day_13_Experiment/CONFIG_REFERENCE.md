# Day 13 Configuration Reference

## Global Hyperparameters

```yaml
preprocessing:
  wiener_window_size: 5
  sampling_rate_in: 200
  sampling_rate_out: 400

segmentation:
  otsu_bins: 256
  min_speech_duration_ms: 100

stag:
  lgbm_trees: 300
  lgbm_max_depth: 7

branch_a:
  cnn_channels: [32, 64, 128]
  blstm_hidden: 256
  gru_hidden: 256
  attention_type: "dot"

branch_b:
  spectrogram_size: [244, 244]
  densenet_growth_rate: 32
```
*Note: YAML representation serves as the target structure for python `dataclass` configs.*
