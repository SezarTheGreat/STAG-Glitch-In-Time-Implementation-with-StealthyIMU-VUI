"""
Day 14 AccEar Experiment - Signal-Level Advanced DSP Filtering Module.

This module provides signal post-filtering methods for AccEar reconstructed spectrograms:
1. Adaptive Wiener Filter (scipy.signal.wiener) for frame-by-frame noise variance reduction.
2. Savitzky-Golay (SG) Filter (scipy.signal.savgol_filter) for time-axis polynomial smoothing.
3. Signal DSP Pipeline combining Adaptive Wiener and Savitzky-Golay filtering.
4. Sample Exporter for generating and saving spectrogram comparison plots and audio WAV files.
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional, Tuple, List, Any

import numpy as np
import scipy.signal
import scipy.io.wavfile
import torch

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for figure rendering
import matplotlib.pyplot as plt

# Import reconstruction helper from data_and_models
try:
    from Day_14_Experiment_AccEar.src.data_and_models import reconstruct_spectrogram
except ImportError:
    from src.data_and_models import reconstruct_spectrogram

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------------------------------
# 1. Adaptive Wiener Filter
# ----------------------------------------------------------------------
def adaptive_wiener_filter(
    spectrogram: Union[np.ndarray, torch.Tensor],
    mysize: Optional[Union[int, Tuple[int, ...]]] = None,
    noise: Optional[float] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies adaptive Wiener filter (scipy.signal.wiener) for frame-by-frame 2D noise
    variance reduction on reconstructed speech spectrograms.

    Supports 2D (H, W), 3D (C, H, W), or 4D (B, C, H, W) NumPy arrays or PyTorch Tensors.
    """
    is_tensor = isinstance(spectrogram, torch.Tensor)
    if is_tensor:
        orig_device = spectrogram.device
        orig_dtype = spectrogram.dtype
        spec_np = spectrogram.detach().cpu().numpy()
    else:
        spec_np = np.asarray(spectrogram)

    shape = spec_np.shape
    ndim = spec_np.ndim

    if ndim == 2:
        filtered_np = scipy.signal.wiener(spec_np, mysize=mysize, noise=noise)
    elif ndim == 3:
        filtered_np = np.empty_like(spec_np)
        for c in range(shape[0]):
            filtered_np[c] = scipy.signal.wiener(spec_np[c], mysize=mysize, noise=noise)
    elif ndim == 4:
        filtered_np = np.empty_like(spec_np)
        for b in range(shape[0]):
            for c in range(shape[1]):
                filtered_np[b, c] = scipy.signal.wiener(spec_np[b, c], mysize=mysize, noise=noise)
    else:
        filtered_np = scipy.signal.wiener(spec_np, mysize=mysize, noise=noise)

    # Spectrogram magnitude values must remain non-negative
    filtered_np = np.clip(filtered_np, 0.0, None)

    if is_tensor:
        return torch.tensor(filtered_np, dtype=orig_dtype, device=orig_device)
    return filtered_np


# ----------------------------------------------------------------------
# 2. Savitzky-Golay Filter
# ----------------------------------------------------------------------
def savitzky_golay_filter(
    spectrogram: Union[np.ndarray, torch.Tensor],
    window_length: int = 7,
    polyorder: int = 2,
    axis: int = -1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies Savitzky-Golay filter (scipy.signal.savgol_filter) along the time axis (axis=-1)
    to smooth step reconstruction artifacts while preserving formant peak sharpness.

    Supports 2D (H, W), 3D (C, H, W), or 4D (B, C, H, W) NumPy arrays or PyTorch Tensors.
    """
    is_tensor = isinstance(spectrogram, torch.Tensor)
    if is_tensor:
        orig_device = spectrogram.device
        orig_dtype = spectrogram.dtype
        spec_np = spectrogram.detach().cpu().numpy()
    else:
        spec_np = np.asarray(spectrogram)

    # Safely adapt window_length and polyorder if time axis dimension is smaller
    axis_size = spec_np.shape[axis]
    effective_window = window_length
    if effective_window > axis_size:
        effective_window = axis_size if axis_size % 2 != 0 else max(1, axis_size - 1)

    effective_poly = polyorder
    if effective_poly >= effective_window:
        effective_poly = max(1, effective_window - 1)

    filtered_np = scipy.signal.savgol_filter(
        spec_np,
        window_length=effective_window,
        polyorder=effective_poly,
        axis=axis
    )

    # Spectrogram magnitude values must remain non-negative
    filtered_np = np.clip(filtered_np, 0.0, None)

    if is_tensor:
        return torch.tensor(filtered_np, dtype=orig_dtype, device=orig_device)
    return filtered_np


# ----------------------------------------------------------------------
# 3. Combined DSP Pipeline
# ----------------------------------------------------------------------
def apply_dsp_pipeline(
    spectrogram: Union[np.ndarray, torch.Tensor],
    wiener_size: Optional[Union[int, Tuple[int, ...]]] = None,
    wiener_noise: Optional[float] = None,
    sg_window_length: int = 7,
    sg_polyorder: int = 2,
    sg_axis: int = -1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Executes sequential signal-level DSP filtering pipeline:
    Step 1: Adaptive Wiener Filter (frame-by-frame 2D noise attenuation)
    Step 2: Savitzky-Golay Filter (time-axis polynomial peak preservation & smoothing)
    """
    wiener_out = adaptive_wiener_filter(spectrogram, mysize=wiener_size, noise=wiener_noise)
    sg_out = savitzky_golay_filter(
        wiener_out,
        window_length=sg_window_length,
        polyorder=sg_polyorder,
        axis=sg_axis
    )
    return sg_out


# ----------------------------------------------------------------------
# 4. Spectrogram-to-Audio Waveform Reconstruction Helper
# ----------------------------------------------------------------------
def spectrogram_to_waveform(
    spectrogram: Union[np.ndarray, torch.Tensor],
    fs: int = 1000,
    n_fft: int = 254,
    hop_length: int = 31,
    n_iter: int = 32
) -> np.ndarray:
    """
    Reconstructs 1D audio time-series waveform from 2D magnitude spectrogram
    using the Griffin-Lim algorithm.
    """
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.detach().cpu().numpy()

    spec_2d = np.squeeze(spectrogram)
    if spec_2d.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram after squeezing, got shape {spec_2d.shape}")

    # Ensure shape is (freq_bins=128, time_frames=128)
    if spec_2d.shape[0] != 128 and spec_2d.shape[1] == 128:
        spec_2d = spec_2d.T

    # Undo STFT magnitude normalization scaling (squared magnitude)
    mag = np.square(np.clip(spec_2d, 0.0, 1.0))

    # Initialize random phase
    angles = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    complex_spec = mag * angles

    nperseg = n_fft
    noverlap = n_fft - hop_length

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_iter):
            _, waveform = scipy.signal.istft(
                complex_spec,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                boundary=None
            )
            _, _, Zxx = scipy.signal.stft(
                waveform,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                boundary=None,
                padded=True
            )

            if Zxx.shape != mag.shape:
                min_h = min(Zxx.shape[0], mag.shape[0])
                min_w = min(Zxx.shape[1], mag.shape[1])
                Zxx_crop = Zxx[:min_h, :min_w]
                mag_crop = mag[:min_h, :min_w]
                angles = np.exp(1j * np.angle(Zxx_crop))
                complex_spec = mag_crop * angles
            else:
                angles = np.exp(1j * np.angle(Zxx))
                complex_spec = mag * angles

        # Final ISTFT
        _, waveform = scipy.signal.istft(
            complex_spec,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None
        )

    max_val = np.max(np.abs(waveform))
    if max_val > 1e-8:
        waveform = waveform / max_val * 0.95

    return waveform.astype(np.float32)


# ----------------------------------------------------------------------
# 5. Save Filtered Samples Exporter
# ----------------------------------------------------------------------
def save_filtered_samples(
    generator: Any,
    dataset: Any,
    output_dir: Union[str, Path],
    num_samples: int = 10,
    device: str = "cpu"
) -> List[Path]:
    """
    Runs AccEar generator on test samples from dataset, applies DSP filtering pipeline,
    and exports reconstructed spectrogram PNG plots and audio WAV files into output_dir.

    Returns list of saved output file paths.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[Path] = []
    num_eval = min(num_samples, len(dataset))
    logging.info(f"Exporting {num_eval} filtered sample reconstructions to {target_dir}")

    for idx in range(num_eval):
        sample = dataset[idx]
        sample_id = sample.get("id", f"sample_{idx+1}")
        transcript = sample.get("transcript", "N/A")
        imu_stft = sample["imu_stft"]  # (1, 128, 128)

        # 1. AccEar Generator Spectrogram Reconstruction
        raw_recon_tensor = reconstruct_spectrogram(generator, imu_stft, device=device)  # (1, 1, 128, 128)

        # 2. Apply DSP Filtering Pipeline
        filt_recon_tensor = apply_dsp_pipeline(raw_recon_tensor)  # (1, 1, 128, 128)

        # Extract 2D numpy arrays (128, 128)
        imu_2d = imu_stft.squeeze().cpu().numpy()
        raw_2d = raw_recon_tensor.squeeze().cpu().numpy()
        filt_2d = filt_recon_tensor.squeeze().cpu().numpy()

        # 3. Audio Waveform Synthesis
        raw_audio = spectrogram_to_waveform(raw_2d, fs=1000)
        filt_audio = spectrogram_to_waveform(filt_2d, fs=1000)

        # File paths
        raw_wav_path = target_dir / f"sample_{idx+1:02d}_{sample_id}_raw.wav"
        filt_wav_path = target_dir / f"sample_{idx+1:02d}_{sample_id}_filtered.wav"
        png_path = target_dir / f"sample_{idx+1:02d}_{sample_id}_spectrogram.png"
        npy_path = target_dir / f"sample_{idx+1:02d}_{sample_id}_filtered_spec.npy"

        # Save WAV files
        scipy.io.wavfile.write(raw_wav_path, 1000, (raw_audio * 32767).astype(np.int16))
        scipy.io.wavfile.write(filt_wav_path, 1000, (filt_audio * 32767).astype(np.int16))

        # Save NPY array
        np.save(npy_path, filt_2d)

        # Save Spectrogram Plot PNG
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        im0 = axes[0].imshow(imu_2d, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Input IMU STFT")
        axes[0].set_xlabel("Time Frame")
        axes[0].set_ylabel("Frequency Bin")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(raw_2d, aspect="auto", origin="lower", cmap="viridis")
        axes[1].set_title("AccEar Raw Reconstruction")
        axes[1].set_xlabel("Time Frame")
        axes[1].set_ylabel("Frequency Bin")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(filt_2d, aspect="auto", origin="lower", cmap="viridis")
        axes[2].set_title("AccEar + DSP Filtered")
        axes[2].set_xlabel("Time Frame")
        axes[2].set_ylabel("Frequency Bin")
        plt.colorbar(im2, ax=axes[2])

        short_transcript = transcript[:45] + "..." if len(transcript) > 45 else transcript
        fig.suptitle(f"Sample {idx+1}/{num_eval} | ID: {sample_id} | Transcript: '{short_transcript}'", fontsize=11)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close(fig)

        saved_files.extend([raw_wav_path, filt_wav_path, npy_path, png_path])

    logging.info(f"Successfully generated {len(saved_files)} files in {target_dir}")
    return saved_files
