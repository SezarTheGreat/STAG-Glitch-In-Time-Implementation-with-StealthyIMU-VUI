import dataclasses

@dataclasses.dataclass
class PreprocessingConfig:
    wiener_window_size: int = 5
    sampling_rate_in: int = 200
    sampling_rate_out: int = 400
    median_filter_kernel_size: int = 5

@dataclasses.dataclass
class SegmentationConfig:
    otsu_bins: int = 256
    min_speech_duration_ms: int = 100
    energy_smooth_kernel: int = 15
    fill_gaps_ms: int = 50

@dataclasses.dataclass
class NormalizationConfig:
    epsilon: float = 1e-8
    use_robust_scaling: bool = False

@dataclasses.dataclass
class StagConfig:
    lgbm_trees: int = 300
    lgbm_max_depth: int = 7
    window_size: int = 2
    target_axis: int = 2  # Accel Z as default acoustic target

@dataclasses.dataclass
class BranchAConfig:
    input_dim: int = 1
    cnn_channels: tuple = (32, 64, 128)
    blstm_hidden: int = 256
    gru_hidden: int = 256
    vocab_size: int = 5000
    dropout: float = 0.3

@dataclasses.dataclass
class BranchBConfig:
    num_classes: int = 50
    spectrogram_size: tuple = (244, 244)
    n_fft: int = 512
    hop_length: int = 8
    n_mels: int = 244
    sample_rate: int = 400
