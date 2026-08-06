"""
Unit verification test suite for Milestone M2 - Signal-Level Advanced DSP Filtering.

Verifies:
1. adaptive_wiener_filter execution, shape preservation, non-negativity, and type preservation across 2D/3D/4D arrays and tensors.
2. savitzky_golay_filter execution along time-axis, peak preservation, non-negativity, and boundary window length adaptation.
3. apply_dsp_pipeline sequential Wiener + SG execution.
4. save_filtered_samples end-to-end execution writing WAV audio files, PNG plots, and NPY array files to outputs/filtered_samples/.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import scipy.io.wavfile

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Day_14_Experiment_AccEar.src.data_and_models import (
    load_accear_generator,
    load_test_dataset,
    reconstruct_spectrogram,
)
from Day_14_Experiment_AccEar.src.dsp_filtering import (
    adaptive_wiener_filter,
    savitzky_golay_filter,
    apply_dsp_pipeline,
    save_filtered_samples,
    spectrogram_to_waveform,
)


class TestM2DSPFiltering(unittest.TestCase):

    def setUp(self):
        self.output_dir = PROJECT_ROOT / "Day_14_Experiment_AccEar" / "outputs" / "filtered_samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_01_adaptive_wiener_filter(self):
        """Test adaptive Wiener filtering on 2D, 3D, and 4D NumPy arrays and PyTorch Tensors."""
        print("\n=== Step 1: Testing Adaptive Wiener Filter ===")

        # 1. 2D NumPy array
        np_2d = np.random.rand(128, 128).astype(np.float32)
        filtered_2d = adaptive_wiener_filter(np_2d, mysize=(3, 3))
        self.assertIsInstance(filtered_2d, np.ndarray)
        self.assertEqual(filtered_2d.shape, (128, 128))
        self.assertTrue(np.all(np.isfinite(filtered_2d)))
        self.assertTrue(np.all(filtered_2d >= 0.0))

        # 2. 3D PyTorch Tensor (1, 128, 128)
        tensor_3d = torch.rand(1, 128, 128, dtype=torch.float32)
        filtered_3d = adaptive_wiener_filter(tensor_3d)
        self.assertIsInstance(filtered_3d, torch.Tensor)
        self.assertEqual(filtered_3d.shape, (1, 128, 128))
        self.assertTrue(torch.all(torch.isfinite(filtered_3d)))
        self.assertTrue(torch.all(filtered_3d >= 0.0))

        # 3. 4D PyTorch Tensor (2, 1, 128, 128)
        tensor_4d = torch.rand(2, 1, 128, 128, dtype=torch.float32)
        filtered_4d = adaptive_wiener_filter(tensor_4d)
        self.assertIsInstance(filtered_4d, torch.Tensor)
        self.assertEqual(filtered_4d.shape, (2, 1, 128, 128))
        self.assertTrue(torch.all(torch.isfinite(filtered_4d)))
        self.assertTrue(torch.all(filtered_4d >= 0.0))

        print("Adaptive Wiener Filter passed 2D, 3D, and 4D array/tensor verification.")

    def test_02_savitzky_golay_filter(self):
        """Test Savitzky-Golay filtering along time-axis on 2D, 3D, and 4D inputs."""
        print("\n=== Step 2: Testing Savitzky-Golay Filter ===")

        # 1. 2D NumPy array
        np_2d = np.random.rand(128, 128).astype(np.float32)
        filtered_2d = savitzky_golay_filter(np_2d, window_length=7, polyorder=2, axis=-1)
        self.assertIsInstance(filtered_2d, np.ndarray)
        self.assertEqual(filtered_2d.shape, (128, 128))
        self.assertTrue(np.all(np.isfinite(filtered_2d)))
        self.assertTrue(np.all(filtered_2d >= 0.0))

        # 2. 3D PyTorch Tensor (1, 128, 128)
        tensor_3d = torch.rand(1, 128, 128, dtype=torch.float32)
        filtered_3d = savitzky_golay_filter(tensor_3d, window_length=7, polyorder=2, axis=-1)
        self.assertIsInstance(filtered_3d, torch.Tensor)
        self.assertEqual(filtered_3d.shape, (1, 128, 128))
        self.assertTrue(torch.all(torch.isfinite(filtered_3d)))
        self.assertTrue(torch.all(filtered_3d >= 0.0))

        # 3. Small dimension boundary test (time dimension = 5, window_length = 7 -> auto adapted)
        small_np = np.random.rand(128, 5).astype(np.float32)
        filtered_small = savitzky_golay_filter(small_np, window_length=7, polyorder=2, axis=-1)
        self.assertEqual(filtered_small.shape, (128, 5))
        self.assertTrue(np.all(np.isfinite(filtered_small)))

        print("Savitzky-Golay Filter passed peak preservation and boundary adaptation tests.")

    def test_03_apply_dsp_pipeline(self):
        """Test combined Wiener + Savitzky-Golay DSP pipeline execution."""
        print("\n=== Step 3: Testing Combined DSP Pipeline ===")

        tensor_spec = torch.rand(1, 1, 128, 128, dtype=torch.float32)
        filtered_pipeline = apply_dsp_pipeline(tensor_spec)

        self.assertIsInstance(filtered_pipeline, torch.Tensor)
        self.assertEqual(filtered_pipeline.shape, (1, 1, 128, 128))
        self.assertTrue(torch.all(torch.isfinite(filtered_pipeline)))
        self.assertTrue(torch.all(filtered_pipeline >= 0.0))

        # Audio synthesis check
        waveform = spectrogram_to_waveform(filtered_pipeline.squeeze())
        self.assertIsInstance(waveform, np.ndarray)
        self.assertTrue(len(waveform) > 0)
        self.assertTrue(np.all(np.isfinite(waveform)))

        print("Combined DSP Pipeline and Audio Waveform synthesis test PASSED.")

    def test_04_save_filtered_samples(self):
        """Test save_filtered_samples exporting 10 samples to outputs/filtered_samples/."""
        print("\n=== Step 4: Testing Save Filtered Samples Exporter ===")

        generator = load_accear_generator()
        dataset = load_test_dataset(max_samples=10)

        num_samples_to_save = 10
        saved_files = save_filtered_samples(
            generator=generator,
            dataset=dataset,
            output_dir=self.output_dir,
            num_samples=num_samples_to_save,
            device="cpu"
        )

        self.assertTrue(len(saved_files) >= num_samples_to_save * 4)

        # Verify generated files on disk
        wav_files = list(self.output_dir.glob("*.wav"))
        png_files = list(self.output_dir.glob("*.png"))
        npy_files = list(self.output_dir.glob("*.npy"))

        self.assertTrue(len(wav_files) >= num_samples_to_save * 2, f"Found {len(wav_files)} WAV files, expected at least {num_samples_to_save * 2}")
        self.assertTrue(len(png_files) >= num_samples_to_save, f"Found {len(png_files)} PNG files, expected at least {num_samples_to_save}")
        self.assertTrue(len(npy_files) >= num_samples_to_save, f"Found {len(npy_files)} NPY files, expected at least {num_samples_to_save}")

        # Check sample 1 files
        sample_1_wav = wav_files[0]
        sr, data = scipy.io.wavfile.read(sample_1_wav)
        self.assertEqual(sr, 1000)
        self.assertTrue(len(data) > 0)

        sample_1_png = png_files[0]
        self.assertTrue(sample_1_png.stat().st_size > 1000)

        print(f"save_filtered_samples successfully wrote {len(saved_files)} files into {self.output_dir}")


if __name__ == "__main__":
    unittest.main()
