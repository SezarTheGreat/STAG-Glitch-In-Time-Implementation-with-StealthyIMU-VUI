"""
Day 14 AccEar Experiment - Data and Model Ingestion Module.

This module provides loaders for:
1. AccEar cGAN Generator model checkpoint (Day_13_Experiment_AccEar/checkpoints/accear_cgan_best_model.pt).
2. StealthyIMU SLU Teacher model checkpoint (CKPT+epoch_30) & SentencePiece Tokenizer (51_unigram.model).
3. 3,070-sentence StealthyIMU test dataset loader and STFT preprocessor.
"""

import os
import sys
import csv
import ast
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List

import numpy as np
import scipy.signal
import scipy.ndimage
import torch
import torch.nn as nn
import sentencepiece as spm

# ----------------------------------------------------------------------
# Paths Configuration & Fallbacks
# ----------------------------------------------------------------------
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent

DEFAULT_ACCEAR_CKPT = WORKSPACE_DIR / "Day_13_Experiment_AccEar" / "checkpoints" / "accear_cgan_best_model.pt"

DEFAULT_TEACHER_CKPT = (
    WORKSPACE_DIR / "models" / "stealthy_imu" / "phase1" / "project" / "results" / "slu_baseline_paper" / "1235" / "save" / "CKPT+epoch_30" / "model.ckpt"
)
FALLBACK_TEACHER_CKPT = (
    WORKSPACE_DIR / "day_04_05_stag_recreation" / "results" / "slu_baseline_paper" / "1235" / "save" / "CKPT+epoch_30" / "model.ckpt"
)

DEFAULT_TOKENIZER_PATH = (
    WORKSPACE_DIR / "models" / "stealthy_imu" / "phase1" / "project" / "pretrain" / "51_unigram.model"
)
FALLBACK_TOKENIZER_PATH = (
    WORKSPACE_DIR / "day_04_05_stag_recreation" / "pretrain" / "51_unigram.model"
)

DEFAULT_HPARAMS_PATH = WORKSPACE_DIR / "day_04_05_stag_recreation" / "paper_exact.yaml"
DEFAULT_TEST_CSV = WORKSPACE_DIR / "day_04_05_stag_recreation" / "results" / "slu_baseline_paper" / "1235" / "test-type=direct.csv"
DEFAULT_DATASET_DIR = WORKSPACE_DIR / "common" / "data" / "StealthyIMU_dataset"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------------------------------
# 1. AccEar cGAN Generator Architecture & Loader
# ----------------------------------------------------------------------
class UNetGenerator(nn.Module):
    """
    AccEar U-Net Generator for IMU-to-Speech Spectrogram Reconstruction.
    Input: (B, 2, 128, 128) - Concatenated noise tensor z ~ N(0,1) and IMU STFT magnitude y.
    Output: (B, 1, 128, 128) - Reconstructed speech spectrogram x_hat in [0, 1].
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 1, features: int = 64):
        super().__init__()

        # Encoder (Downsampling)
        # 128x128 -> 64x64
        self.enc1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1)
        # 64x64 -> 32x32
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
        )
        # 32x32 -> 16x16
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
        )
        # 16x16 -> 8x8
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 8),
        )
        # 8x8 -> 4x4 (Bottleneck)
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder (Upsampling with Skip Connections)
        # 4x4 -> 8x8
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
        )
        # 8x8 -> 16x16 (Concat skip enc4 -> 512 + 512 = 1024 channels)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(features * 16, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
        )
        # 16x16 -> 32x32 (Concat skip enc3 -> 256 + 256 = 512 channels)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
        )
        # 32x32 -> 64x64 (Concat skip enc2 -> 128 + 128 = 256 channels)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        # 64x64 -> 128x128 (Concat skip enc1 -> 64 + 64 = 128 channels)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_or_x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for AccEar U-Net generator.
        Accepts either separate z and y tensors or a single 2-channel tensor (B, 2, 128, 128).
        """
        if y is not None:
            x_in = torch.cat([z_or_x, y], dim=1)
        else:
            x_in = z_or_x

        e1 = self.enc1(x_in)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        d4 = self.dec4(b)
        d4_cat = torch.cat([d4, e4], dim=1)

        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2_cat)
        d1_cat = torch.cat([d1, e1], dim=1)

        x_hat = self.final_up(d1_cat)
        return x_hat


def load_accear_generator(
    ckpt_path: Optional[Union[str, Path]] = None,
    device: str = "cpu"
) -> UNetGenerator:
    """
    Loads pre-trained AccEar cGAN Generator from checkpoint.
    """
    resolved_path = Path(ckpt_path) if ckpt_path else DEFAULT_ACCEAR_CKPT
    if not resolved_path.exists():
        raise FileNotFoundError(f"AccEar generator checkpoint not found at: {resolved_path}")

    model = UNetGenerator(in_channels=2, out_channels=1, features=64).to(device)
    ckpt = torch.load(resolved_path, map_location=device)

    if "generator_state_dict" in ckpt:
        state_dict = ckpt["generator_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    logging.info(f"Loaded AccEar cGAN Generator from {resolved_path} onto {device}")
    return model


def reconstruct_spectrogram(
    generator: UNetGenerator,
    imu_stft: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Generates reconstructed spectrogram (B, 1, 128, 128) from IMU STFT spectrogram tensor (B, 1, 128, 128) or (128, 128).
    """
    generator.eval()
    if imu_stft.dim() == 2:
        imu_stft = imu_stft.unsqueeze(0).unsqueeze(0)
    elif imu_stft.dim() == 3:
        imu_stft = imu_stft.unsqueeze(0)  # (1, 1, 128, 128)

    imu_stft = imu_stft.to(device)
    if z is None:
        z = torch.randn_like(imu_stft).to(device)
    else:
        if z.dim() == 2:
            z = z.unsqueeze(0).unsqueeze(0)
        elif z.dim() == 3:
            z = z.unsqueeze(0)
        z = z.to(device)

    with torch.no_grad():
        reconstructed = generator(z, imu_stft)
    return reconstructed


import pickle
import scipy.interpolate as interpolate

DEFAULT_UPSCALER_PATH = WORKSPACE_DIR / "models" / "stealthy_imu" / "upscaler.pkl"
FALLBACK_UPSCALER_PATH = WORKSPACE_DIR / "day_04_05_stag_recreation" / "models" / "upscaler.pkl"

day13_src_path = str(WORKSPACE_DIR / "Day_13_Experiment" / "src")
if day13_src_path not in sys.path:
    sys.path.insert(0, day13_src_path)

stag_src_path = str(WORKSPACE_DIR / "day_04_05_stag_recreation")
if stag_src_path not in sys.path:
    sys.path.insert(0, stag_src_path)

try:
    import module1_capture_denoise as m1
    import module2_segmentation as m2
    import module3_normalization as m3
    from config import PreprocessingConfig, SegmentationConfig, NormalizationConfig
    from feature import AccSpec
except ImportError:
    m1 = None
    m2 = None
    m3 = None
    AccSpec = None


def load_day13_upscaler():
    path = DEFAULT_UPSCALER_PATH if DEFAULT_UPSCALER_PATH.exists() else FALLBACK_UPSCALER_PATH
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def reconstruct_day13_hybrid_features(
    acc_path: Union[str, Path],
    duration: float,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Reconstructs 31-bin STFT features (1, T, 31) using Day 13 Hybrid Architecture:
    InertiEAR VAD + Active-Speech Z-Score Scaling + STAG LightGBM Upscaler + AccSpec.
    """
    acc_path = Path(acc_path)
    if not acc_path.exists():
        return torch.zeros(1, 100, 31, device=device)

    acc_data = np.load(acc_path)
    if acc_data.ndim == 2 and acc_data.shape[0] == 4:
        acc_odd = acc_data[3]
        gyro_even = acc_data[1:4]
    elif acc_data.ndim == 2 and acc_data.shape[0] >= 3:
        acc_odd = acc_data[2]
        gyro_even = acc_data[0:3]
    else:
        acc_odd = acc_data.flatten()
        gyro_even = np.zeros((3, len(acc_odd)))

    if m1 is not None and m2 is not None and m3 is not None:
        prep_cfg = PreprocessingConfig()
        seg_cfg = SegmentationConfig()
        norm_cfg = NormalizationConfig()

        acc_denoised, gyro_denoised = m1.process_raw_imu(acc_odd.reshape(1, -1), gyro_even, prep_cfg)
        speech_mask = m2.segment_speech(acc_denoised, gyro_denoised, prep_cfg, seg_cfg)
        features_denoised = np.vstack([acc_denoised, gyro_denoised])
        features_norm = m3.apply_device_independent_scaling(features_denoised, speech_mask, norm_cfg)
        acc_norm = features_norm[0, :]
    else:
        acc_norm = zero_mean_normalize(acc_odd)

    t_source = np.linspace(0.0, duration, len(acc_norm), endpoint=False)
    t_target = np.linspace(0.0, duration, int(duration * 500), endpoint=False)
    f_interp = interpolate.interp1d(t_source, acc_norm, kind='cubic', fill_value='extrapolate')
    recon_500 = f_interp(t_target)

    nyq = 0.5 * 500.0
    b, a = scipy.signal.butter(4, 80.0 / nyq, btype="low")
    recon_lp = scipy.signal.filtfilt(b, a, recon_500)

    if AccSpec is not None:
        acc_spec_extractor = AccSpec(sample_rate=500, n_fft=80, win_length=80, hop_length=20).to(device)
        wav_tensor = torch.from_numpy(recon_lp.copy()).float().unsqueeze(0).to(device)
        feats = acc_spec_extractor(wav_tensor)
        return feats
    else:
        return torch.from_numpy(recon_lp.copy()).float().unsqueeze(0).unsqueeze(-1).repeat(1, 1, 31).to(device)


# ----------------------------------------------------------------------
# 2. StealthyIMU SLU Teacher Model & Tokenizer Loader
# ----------------------------------------------------------------------
class TeacherSLUModel(nn.Module):
    """
    Wrapper for StealthyIMU SpeechBrain SLU Teacher Model.
    Exposes CRDNN encoder, AttentionalRNNDecoder, output embedding, and linear projection modules.
    """

    def __init__(self, modules_dict: nn.ModuleDict, tokenizer: spm.SentencePieceProcessor, hparams: Dict[str, Any]):
        super().__init__()
        self.modules_dict = modules_dict
        self.tokenizer = tokenizer
        self.hparams = hparams

    def _prepare_input_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensures input spectrogram x has shape (B, T, 31) matching CRDNN n_feature=31.
        Supports 2D (T, F), 3D (B, T, F), and 4D (B, C, T, F) inputs.
        Mapped to the 61 Hz - 250 Hz STFT frequency band expected by Teacher SLU.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, T, F)
        elif x.dim() == 4:
            # (B, 1, H, W) or (B, C, T, F) -> (B, T, F)
            x = x.squeeze(1)

        if x.size(-1) == 128:
            # Map 128-bin linear STFT (0..500 Hz) to the 31-bin band (61..250 Hz) expected by AccSpec (bins 15..46)
            x = x[:, :, 15:46]
        elif x.size(-1) > 31:
            x = x[:, :, :31]
        elif x.size(-1) < 31:
            pad_size = 31 - x.size(-1)
            x = torch.nn.functional.pad(x, (0, pad_size))

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs encoder forward pass on spectrogram input tensor (B, T, 31) or (B, 1, 128, 128).
        Returns encoded features (B, T_downsampled, Encoder_Dim).
        """
        x_prep = self._prepare_input_features(x)
        enc_out = self.modules_dict["enc"](x_prep)
        return enc_out

    def predict(self, spectrogram: torch.Tensor) -> List[str]:
        """
        Full sequence greedy/beam decoding is scheduled for Milestone M3 (Features 7-9).
        Raises NotImplementedError in Milestone M1 to eliminate facade implementations.
        """
        raise NotImplementedError("TeacherSLUModel.predict() sequence decoding is scheduled for Milestone M3.")



def load_slu_teacher_and_tokenizer(
    ckpt_path: Optional[Union[str, Path]] = None,
    tokenizer_path: Optional[Union[str, Path]] = None,
    hparams_path: Optional[Union[str, Path]] = None,
    device: str = "cpu"
) -> Tuple[TeacherSLUModel, spm.SentencePieceProcessor]:
    """
    Loads pre-trained StealthyIMU SLU Teacher model (CKPT+epoch_30) and SentencePiece tokenizer.
    """
    # 1. Resolve paths with fallbacks
    resolved_ckpt = Path(ckpt_path) if ckpt_path else DEFAULT_TEACHER_CKPT
    if not resolved_ckpt.exists() and FALLBACK_TEACHER_CKPT.exists():
        resolved_ckpt = FALLBACK_TEACHER_CKPT

    resolved_tok = Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER_PATH
    if not resolved_tok.exists() and FALLBACK_TOKENIZER_PATH.exists():
        resolved_tok = FALLBACK_TOKENIZER_PATH

    resolved_hp = Path(hparams_path) if hparams_path else DEFAULT_HPARAMS_PATH

    if not resolved_ckpt.exists():
        raise FileNotFoundError(f"Teacher model checkpoint not found at: {resolved_ckpt}")
    if not resolved_tok.exists():
        raise FileNotFoundError(f"Tokenizer model not found at: {resolved_tok}")
    if not resolved_hp.exists():
        raise FileNotFoundError(f"Hparams YAML not found at: {resolved_hp}")

    # 2. Safety Monkeypatch for SpeechBrain / k2 dependency
    from unittest.mock import MagicMock
    if "k2" not in sys.modules or isinstance(sys.modules.get("k2"), MagicMock):
        sys.modules["k2"] = MagicMock()

    import speechbrain.utils.importutils as iu
    if not getattr(iu.LazyModule, "_is_patched", False):
        _orig_getattr = iu.LazyModule.__getattr__
        def _safe_getattr(self, attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return _orig_getattr(self, attr)
        iu.LazyModule.__getattr__ = _safe_getattr
        iu.LazyModule._is_patched = True

    # Ensure day_04_05_stag_recreation is in sys.path for custom features/models
    stag_dir = str(WORKSPACE_DIR / "day_04_05_stag_recreation")
    if stag_dir not in sys.path:
        sys.path.append(stag_dir)

    from hyperpyyaml import load_hyperpyyaml

    # 3. Load Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(resolved_tok))
    logging.info(f"Loaded SentencePiece Tokenizer from {resolved_tok} (vocab size = {tokenizer.vocab_size()})")

    # 4. Load Hparams & SpeechBrain Modules
    with open(resolved_hp, "r", encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f, overrides={
            "output_folder": str(resolved_ckpt.parent.parent),
            "tokenizer_file": str(resolved_tok),
        })

    raw_state_dict = torch.load(resolved_ckpt, map_location=device)

    # 5. Map state_dict keys from SpeechBrain Checkpoint to ModuleDict
    prefix_map = {"0.": "enc.", "1.": "output_emb.", "2.": "dec.", "3.": "seq_lin."}
    mapped_state_dict = {}
    for k, v in raw_state_dict.items():
        if "num_batches_tracked" in k:
            continue
        for p_old, p_new in prefix_map.items():
            if k.startswith(p_old):
                mapped_state_dict[p_new + k[len(p_old):]] = v
                break

    modules_dict = nn.ModuleDict(hparams["modules"]).to(device)
    missing, unexpected = modules_dict.load_state_dict(mapped_state_dict, strict=False)
    if missing:
        logging.warning(f"Teacher state dict missing keys: {missing}")
    if unexpected:
        logging.warning(f"Teacher state dict unexpected keys: {unexpected}")

    modules_dict.eval()
    logging.info(f"Loaded StealthyIMU Teacher Model from {resolved_ckpt} onto {device}")

    model_wrapper = TeacherSLUModel(modules_dict, tokenizer, hparams)
    return model_wrapper, tokenizer


# ----------------------------------------------------------------------
# 3. Preprocessing Helper Functions
# ----------------------------------------------------------------------
def zero_mean_normalize(signal: np.ndarray) -> np.ndarray:
    """Applies zero-mean unit-variance normalization to 1D signal."""
    mean_val = np.mean(signal)
    std_val = np.std(signal) + 1e-8
    return (signal - mean_val) / std_val


def highpass_filter_20hz(signal: np.ndarray, fs: float = 1000.0, order: int = 4) -> np.ndarray:
    """Applies 4th-order Butterworth high-pass filter at 20 Hz."""
    nyquist = 0.5 * fs
    cutoff = 20.0 / nyquist
    if cutoff >= 1.0:
        cutoff = 0.99
    b, a = scipy.signal.butter(order, cutoff, btype="high", analog=False)
    return scipy.signal.filtfilt(b, a, signal)


def linear_interpolate_1khz(
    raw_timestamps: Optional[np.ndarray],
    raw_signal: np.ndarray,
    duration_sec: float = 4.0
) -> np.ndarray:
    """Resamples raw IMU signal to 1 kHz uniform time grid over duration_sec."""
    target_num_samples = int(duration_sec * 1000)
    uniform_timestamps = np.linspace(0.0, duration_sec, target_num_samples, endpoint=False)
    if raw_timestamps is None or len(raw_timestamps) != len(raw_signal):
        raw_timestamps = np.linspace(0.0, duration_sec, len(raw_signal), endpoint=False)
    else:
        raw_timestamps = (raw_timestamps - raw_timestamps[0]) / 1000.0 if raw_timestamps[0] > 1000.0 else raw_timestamps
    return np.interp(uniform_timestamps, raw_timestamps, raw_signal)


def compute_imu_stft_spectrogram(
    imu_signal_1khz: np.ndarray,
    fs: int = 1000,
    n_fft: int = 254,
    hop_length: int = 31,
    target_shape: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Computes STFT magnitude spectrogram from 1 kHz IMU signal and resizes to target shape (128, 128).
    Returns normalized float32 array in [0, 1]. Zero-pads short signals if length < n_fft.
    """
    if len(imu_signal_1khz) < n_fft:
        pad_len = n_fft - len(imu_signal_1khz)
        imu_signal_1khz = np.pad(imu_signal_1khz, (0, pad_len), mode="constant")

    frequencies, times, Zxx = scipy.signal.stft(
        imu_signal_1khz,
        fs=fs,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(Zxx)
    sqrt_mag = np.sqrt(magnitude)
    min_val = np.min(sqrt_mag)
    max_val = np.max(sqrt_mag) + 1e-8
    norm_mag = (sqrt_mag - min_val) / (max_val - min_val)

    h, w = norm_mag.shape
    target_h, target_w = target_shape
    zoom_h = target_h / float(h)
    zoom_w = target_w / float(w)

    norm_mag_resized = scipy.ndimage.zoom(norm_mag, (zoom_h, zoom_w), order=1)
    norm_mag_resized = np.clip(norm_mag_resized[:target_h, :target_w], 0.0, 1.0)
    return norm_mag_resized.astype(np.float32)


# ----------------------------------------------------------------------
# 4. 3,070 Test Dataset Loader & PyTorch Dataset Class
# ----------------------------------------------------------------------
class StealthyIMUTestDataset(torch.utils.data.Dataset):
    """
    Dataset loader for 3,070 StealthyIMU test split items.
    """

    def __init__(
        self,
        metadata_csv: Optional[Union[str, Path]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        strict: bool = True,
    ):
        self.metadata_csv = Path(metadata_csv) if metadata_csv else DEFAULT_TEST_CSV
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        self.strict = strict

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found at: {self.metadata_csv}")
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found at: {self.dataset_dir}")

        self.samples = []
        self._load_metadata()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logging.info(f"Initialized StealthyIMUTestDataset with {len(self.samples)} samples from {self.metadata_csv}")

    def _load_metadata(self):
        with open(self.metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row["ID"]
                duration = float(row["duration"])
                wav_rel = row["wav"]
                transcript = row.get("transcript", "")
                semantics_str = row.get("semantics", "{}")

                # Parse semantics dict string
                try:
                    # Semantics string formatting in CSV uses pipe characters: {'action': 'weather'| 'entities': ...}
                    cleaned_sem = semantics_str.replace("|", ",")
                    semantics_dict = ast.literal_eval(cleaned_sem)
                except Exception:
                    semantics_dict = {"action": "unknown", "entities": []}

                intent = semantics_dict.get("action", "unknown")

                # Resolve accnpy path relative to dataset_dir
                clean_wav_rel = wav_rel.replace("./", "").replace(".wav", ".accnpy")
                accnpy_path = self.dataset_dir / clean_wav_rel

                self.samples.append({
                    "id": uid,
                    "duration": duration,
                    "wav_rel": wav_rel,
                    "accnpy_path": accnpy_path,
                    "transcript": transcript,
                    "intent": intent,
                    "semantics": semantics_dict,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        acc_path = sample["accnpy_path"]
        duration = sample["duration"]

        if not acc_path.exists():
            if self.strict:
                raise FileNotFoundError(f"IMU signal file not found at: {acc_path}")
            else:
                logging.warning(f"File not found: {acc_path}. Returning zero tensor.")
                imu_tensor = torch.zeros(1, 128, 128, dtype=torch.float32)
        else:
            try:
                acc_data = np.load(acc_path)
                if acc_data.ndim == 2 and acc_data.shape[0] == 4:
                    raw_ts = acc_data[0]
                    z_axis = acc_data[3]
                elif acc_data.ndim == 2 and acc_data.shape[0] >= 3:
                    raw_ts = None
                    z_axis = acc_data[2]
                else:
                    raw_ts = None
                    z_axis = acc_data.flatten()

                s_norm = zero_mean_normalize(z_axis)
                s_hp = highpass_filter_20hz(s_norm, fs=1000.0)
                s_1khz = linear_interpolate_1khz(raw_ts, s_hp, duration_sec=duration)
                stft_spec = compute_imu_stft_spectrogram(s_1khz, fs=1000, target_shape=(128, 128))
                imu_tensor = torch.from_numpy(stft_spec).unsqueeze(0)  # (1, 128, 128)
            except Exception as e:
                if self.strict:
                    raise RuntimeError(f"Error processing IMU signal {acc_path}: {e}") from e
                else:
                    logging.warning(f"Error loading {acc_path}: {e}. Returning zero tensor.")
                    imu_tensor = torch.zeros(1, 128, 128, dtype=torch.float32)

        return {
            "id": sample["id"],
            "duration": sample["duration"],
            "transcript": sample["transcript"],
            "intent": sample["intent"],
            "semantics": sample["semantics"],
            "imu_stft": imu_tensor,
            "accnpy_path": str(acc_path),
        }


def stealthy_imu_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for StealthyIMUTestDataset samples.
    Handles collating variable-length semantics dictionary and other metadata fields
    when using PyTorch DataLoader with batch_size > 1.
    """
    ids = [b["id"] for b in batch]
    durations = torch.tensor([b["duration"] for b in batch], dtype=torch.float32)
    transcripts = [b["transcript"] for b in batch]
    intents = [b["intent"] for b in batch]
    semantics = [b["semantics"] for b in batch]
    imu_stfts = torch.stack([b["imu_stft"] for b in batch], dim=0)
    accnpy_paths = [b["accnpy_path"] for b in batch]

    return {
        "id": ids,
        "duration": durations,
        "transcript": transcripts,
        "intent": intents,
        "semantics": semantics,
        "imu_stft": imu_stfts,
        "accnpy_path": accnpy_paths,
    }


def load_test_dataset(
    metadata_csv: Optional[Union[str, Path]] = None,
    dataset_dir: Optional[Union[str, Path]] = None,
    max_samples: Optional[int] = None,
    strict: bool = True,
) -> StealthyIMUTestDataset:
    """
    Helper function to load the 3,070-sentence StealthyIMU test dataset.
    """
    return StealthyIMUTestDataset(
        metadata_csv=metadata_csv,
        dataset_dir=dataset_dir,
        max_samples=max_samples,
        strict=strict,
    )
