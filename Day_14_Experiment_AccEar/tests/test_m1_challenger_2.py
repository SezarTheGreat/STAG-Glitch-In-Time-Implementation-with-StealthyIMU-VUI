"""
Empirical Verification Test Suite by Challenger M1-2.

Focus:
1. Dataset Indexing & Data Integrity across random samples.
2. Model GPU/CPU Device Placement Flexibility.
3. Output Tensor Dtype & Dimensions.
"""

import sys
import os
import unittest
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Day_14_Experiment_AccEar.src.data_and_models import (
    UNetGenerator,
    load_accear_generator,
    reconstruct_spectrogram,
    load_slu_teacher_and_tokenizer,
    TeacherSLUModel,
    StealthyIMUTestDataset,
    load_test_dataset,
    compute_imu_stft_spectrogram,
)


class TestChallengerM1_2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("   CHALLENGER M1-2: EMPIRICAL VERIFICATION SUITE       ")
        print("=======================================================\n")
        print("Pre-loading models and dataset for fast empirical execution...")
        cls.generator = load_accear_generator(device="cpu")
        cls.teacher_model, cls.tokenizer = load_slu_teacher_and_tokenizer(device="cpu")
        cls.dataset = load_test_dataset()

    def test_01_dataset_indexing_and_data_integrity(self):
        """Test dataset indexing across boundary and random samples."""
        print("\n--- Test 1: Dataset Indexing & Data Integrity Across Random Samples ---")
        total_len = len(self.dataset)
        self.assertEqual(total_len, 3070, f"Expected 3070 items in test dataset, got {total_len}")

        # Pick boundary indices and 10 random indices
        np.random.seed(12345)
        random_indices = list(np.random.choice(total_len, size=10, replace=False))
        boundary_indices = [0, 1535, total_len - 1]
        test_indices = sorted(list(set(boundary_indices + random_indices)))

        print(f"Testing {len(test_indices)} distinct dataset indices (including 0, 1535, 3069)...")

        required_keys = {"id", "duration", "transcript", "intent", "semantics", "imu_stft", "accnpy_path"}

        for idx in test_indices:
            sample = self.dataset[idx]
            self.assertEqual(set(sample.keys()), required_keys, f"Sample {idx} missing keys")

            # Check imu_stft tensor properties
            imu_stft = sample["imu_stft"]
            self.assertIsInstance(imu_stft, torch.Tensor, f"Sample {idx} imu_stft must be torch.Tensor")
            self.assertEqual(imu_stft.shape, (1, 128, 128), f"Sample {idx} imu_stft shape mismatch: {imu_stft.shape}")
            self.assertEqual(imu_stft.dtype, torch.float32, f"Sample {idx} imu_stft dtype mismatch: {imu_stft.dtype}")
            self.assertTrue(torch.isfinite(imu_stft).all(), f"Sample {idx} imu_stft contains non-finite values (NaN/Inf)")

            # Check value range [0, 1]
            min_val = imu_stft.min().item()
            max_val = imu_stft.max().item()
            self.assertGreaterEqual(min_val, 0.0 - 1e-6, f"Sample {idx} min value < 0: {min_val}")
            self.assertLessEqual(max_val, 1.0 + 1e-6, f"Sample {idx} max value > 1: {max_val}")

            # Check metadata types and non-emptiness
            self.assertIsInstance(sample["id"], str, f"Sample {idx} ID not str")
            self.assertGreater(len(sample["id"]), 0, f"Sample {idx} ID empty")
            self.assertIsInstance(sample["duration"], float, f"Sample {idx} duration not float")
            self.assertGreater(sample["duration"], 0.0, f"Sample {idx} duration <= 0")
            self.assertIsInstance(sample["transcript"], str, f"Sample {idx} transcript not str")
            self.assertGreater(len(sample["transcript"]), 0, f"Sample {idx} transcript empty")
            self.assertIsInstance(sample["intent"], str, f"Sample {idx} intent not str")
            self.assertNotEqual(sample["intent"], "", f"Sample {idx} intent empty string")
            self.assertIsInstance(sample["semantics"], dict, f"Sample {idx} semantics not dict")

        # Test Out-of-Bounds Indexing
        with self.assertRaises(IndexError, msg="Accessing index out of bounds should raise IndexError"):
            _ = self.dataset[total_len]

        # Test Python negative indexing (-1 = last element)
        last_sample = self.dataset[-1]
        self.assertEqual(last_sample["id"], self.dataset[total_len - 1]["id"])

        print(f"PASSED: Dataset indexing and integrity verified across {len(test_indices)} sample indices.")

    def test_02_model_device_placement_flexibility(self):
        """Test model GPU/CPU device placement flexibility for Generator and Teacher."""
        print("\n--- Test 2: Model GPU/CPU Device Placement Flexibility ---")

        # 1. Test UNetGenerator parameter device
        gen_param_device = next(self.generator.parameters()).device
        self.assertEqual(gen_param_device.type, "cpu", f"Generator parameter device type expected 'cpu', got '{gen_param_device.type}'")

        # 2. Test UNetGenerator with torch.device object
        torch_device_cpu = torch.device("cpu")
        gen_torch_dev = load_accear_generator(device=torch_device_cpu)
        self.assertEqual(next(gen_torch_dev.parameters()).device.type, "cpu")

        # 3. Test TeacherSLUModel parameter device
        teacher_param_device = next(self.teacher_model.parameters()).device
        self.assertEqual(teacher_param_device.type, "cpu")

        # 4. Test tensor device placement during forward pass & reconstruct_spectrogram
        dummy_imu = torch.rand(2, 1, 128, 128, device="cpu")
        recon = reconstruct_spectrogram(self.generator, dummy_imu, device="cpu")
        self.assertEqual(recon.device.type, "cpu")

        # 5. Check CUDA device handling logic
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available on system: {cuda_available}")

        if cuda_available:
            gen_gpu = load_accear_generator(device="cuda")
            self.assertEqual(next(gen_gpu.parameters()).device.type, "cuda")

            teacher_gpu, _ = load_slu_teacher_and_tokenizer(device="cuda")
            self.assertEqual(next(teacher_gpu.parameters()).device.type, "cuda")

            dummy_imu_gpu = torch.rand(2, 1, 128, 128, device="cuda")
            recon_gpu = reconstruct_spectrogram(gen_gpu, dummy_imu_gpu, device="cuda")
            self.assertEqual(recon_gpu.device.type, "cuda")
            print("CUDA device placement tests PASSED.")
        else:
            with self.assertRaises(Exception, msg="Requesting CUDA on non-CUDA machine should raise an exception"):
                _ = load_accear_generator(device="cuda")
            print("Verified CUDA request fails gracefully with exception on CPU-only system.")

        print("PASSED: Model device placement flexibility verified.")

    def test_03_output_tensor_dtype_and_dimensions(self):
        """Test output tensor dtype, shape consistency, and value bounds."""
        print("\n--- Test 3: Output Tensor Dtype and Dimension Consistency ---")

        batch_sizes = [1, 2, 4, 8, 16]

        # 1. Test Generator across different batch sizes
        for B in batch_sizes:
            z = torch.randn(B, 1, 128, 128, dtype=torch.float32)
            y = torch.rand(B, 1, 128, 128, dtype=torch.float32)
            combined = torch.cat([z, y], dim=1)  # (B, 2, 128, 128)

            with torch.no_grad():
                out = self.generator(combined)

            self.assertEqual(out.shape, (B, 1, 128, 128), f"Generator output shape mismatch for batch size {B}: {out.shape}")
            self.assertEqual(out.dtype, torch.float32, f"Generator output dtype mismatch for batch size {B}: {out.dtype}")
            self.assertTrue((out >= 0.0).all() and (out <= 1.0).all(), f"Generator output values not in range [0, 1] for batch size {B}")

            # Test reconstruct_spectrogram helper function
            recon = reconstruct_spectrogram(self.generator, y, z=z, device="cpu")
            self.assertEqual(recon.shape, (B, 1, 128, 128))
            self.assertEqual(recon.dtype, torch.float32)

        print(f"Verified UNetGenerator outputs shape (B, 1, 128, 128) float32 in [0, 1] across batch sizes {batch_sizes}.")

        # 2. Test reconstruct_spectrogram with 3D input tensor (1, 128, 128)
        single_imu_3d = torch.rand(1, 128, 128, dtype=torch.float32)
        recon_3d = reconstruct_spectrogram(self.generator, single_imu_3d, device="cpu")
        self.assertEqual(recon_3d.shape, (1, 1, 128, 128), f"3D input reconstruction shape expected (1, 1, 128, 128), got {recon_3d.shape}")
        self.assertEqual(recon_3d.dtype, torch.float32)

        # 3. Test TeacherSLUModel forward pass across batch sizes and input feature dimensions
        for B in batch_sizes:
            # 4D input tensor (B, 1, 128, 128)
            spec_4d = torch.rand(B, 1, 128, 128, dtype=torch.float32)
            with torch.no_grad():
                feats_4d = self.teacher_model(spec_4d)

            self.assertEqual(feats_4d.dim(), 3, f"Teacher output tensor should be 3D (B, T, D), got {feats_4d.dim()}D")
            self.assertEqual(feats_4d.shape[0], B, f"Teacher output batch size expected {B}, got {feats_4d.shape[0]}")
            self.assertEqual(feats_4d.shape[2], 256, f"Teacher output feature dim expected 256, got {feats_4d.shape[2]}")
            self.assertEqual(feats_4d.dtype, torch.float32, f"Teacher output dtype expected float32, got {feats_4d.dtype}")

            # 3D input tensor (B, T, 31)
            spec_3d = torch.rand(B, 100, 31, dtype=torch.float32)
            with torch.no_grad():
                feats_3d = self.teacher_model(spec_3d)

            self.assertEqual(feats_3d.shape[0], B)
            self.assertEqual(feats_3d.shape[2], 256)
            self.assertEqual(feats_3d.dtype, torch.float32)

        print(f"Verified TeacherSLUModel outputs shape (B, T_downsampled, 256) float32 across batch sizes {batch_sizes}.")

        # 4. Test TeacherSLUModel feature padding/slicing in _prepare_input_features
        # Feature dim < 31 (e.g. 20) -> should pad to 31
        spec_under = torch.rand(2, 50, 20, dtype=torch.float32)
        prep_under = self.teacher_model._prepare_input_features(spec_under)
        self.assertEqual(prep_under.shape, (2, 50, 31), f"Padded input feature shape mismatch: {prep_under.shape}")

        # Feature dim > 31 (e.g. 64) -> should slice to 31
        spec_over = torch.rand(2, 50, 64, dtype=torch.float32)
        prep_over = self.teacher_model._prepare_input_features(spec_over)
        self.assertEqual(prep_over.shape, (2, 50, 31), f"Sliced input feature shape mismatch: {prep_over.shape}")

        print("Verified _prepare_input_features padding and slicing logic.")

        # 5. Check dtype tolerance and float64 input behavior
        # 5a. Generator with 2-channel float64 tensor cast to float32
        spec_double_2ch = torch.rand(2, 2, 128, 128, dtype=torch.float64)
        with torch.no_grad():
            out_float32 = self.generator(spec_double_2ch.to(torch.float32))
        self.assertEqual(out_float32.dtype, torch.float32)

        # 5b. Un-cast float64 tensor in reconstruct_spectrogram -> demonstrates float32 vs float64 behavior
        spec_double_1ch = torch.rand(1, 1, 128, 128, dtype=torch.float64)
        with self.assertRaises(RuntimeError, msg="Passing float64 tensor to float32 model should raise RuntimeError"):
            _ = self.generator(spec_double_1ch, spec_double_1ch)

        print("Verified float64 vs float32 dtype mismatch exception behavior.")
        print("PASSED: Output tensor dtype and dimension consistency verified.")


if __name__ == "__main__":
    unittest.main()
