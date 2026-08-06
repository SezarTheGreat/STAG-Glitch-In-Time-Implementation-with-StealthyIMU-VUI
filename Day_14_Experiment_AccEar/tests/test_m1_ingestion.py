"""
Unit verification script for Milestone M1 - Model & Data Ingestion Setup.

Verifies:
1. AccEar cGAN Generator loader function and inference on dummy/real tensors (1, 2, 128, 128) -> (1, 1, 128, 128).
2. StealthyIMU SLU Teacher Model and SentencePiece unigram tokenizer (51_unigram.model).
3. 3,070 Test Dataset Loader (metadata and 200 Hz IMU traces / STFT representations).
4. End-to-end integration (dataset -> generator -> teacher encoder).
5. TeacherSLUModel.predict() raises NotImplementedError for Milestone M1 (no facade logic).
6. Strict mode error handling for dataset loaders and path validation.
7. Custom stealthy_imu_collate_fn for DataLoader batching with variable-length semantics dict.
8. Tensor rank normalization, short signal padding, and edge case safety.
"""

import sys
import unittest
from pathlib import Path
import torch
import numpy as np

# Ensure parent directory is in sys.path so Day_14_Experiment_AccEar package can be imported
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
    stealthy_imu_collate_fn,
)


class TestM1Ingestion(unittest.TestCase):

    def test_01_accear_generator_loading_and_inference(self):
        """Test AccEar cGAN Generator model loading and inference on dummy tensor."""
        logging_msg = "=== Step 1: Testing AccEar Generator Loading & Inference ==="
        print("\n" + logging_msg)

        generator = load_accear_generator()
        self.assertIsNotNone(generator)
        self.assertIsInstance(generator, UNetGenerator)
        self.assertFalse(generator.training, "Generator must be in evaluation mode.")

        # Test inference with combined input tensor (1, 2, 128, 128)
        dummy_z = torch.randn(1, 1, 128, 128)
        dummy_y = torch.rand(1, 1, 128, 128)
        dummy_combined = torch.cat([dummy_z, dummy_y], dim=1)  # (1, 2, 128, 128)

        with torch.no_grad():
            output_from_combined = generator(dummy_combined)
        self.assertEqual(output_from_combined.shape, (1, 1, 128, 128))
        self.assertTrue(torch.isfinite(output_from_combined).all())

        # Test helper function reconstruct_spectrogram
        recon_spec = reconstruct_spectrogram(generator, dummy_y, z=dummy_z)
        self.assertEqual(recon_spec.shape, (1, 1, 128, 128))
        self.assertTrue((recon_spec >= 0.0).all() and (recon_spec <= 1.0).all())

        print("AccEar Generator loaded and executed inference successfully. Output shape: (1, 1, 128, 128)")

    def test_02_slu_teacher_and_tokenizer_loading(self):
        """Test StealthyIMU Teacher Model & SentencePiece Tokenizer loading."""
        logging_msg = "=== Step 2: Testing StealthyIMU Teacher Model & Tokenizer Loading ==="
        print("\n" + logging_msg)

        teacher_model, tokenizer = load_slu_teacher_and_tokenizer()
        self.assertIsNotNone(teacher_model)
        self.assertIsNotNone(tokenizer)
        self.assertIsInstance(teacher_model, TeacherSLUModel)

        # Verify Tokenizer
        vocab_size = tokenizer.vocab_size()
        self.assertEqual(vocab_size, 51, f"Expected vocabulary size 51, got {vocab_size}")

        test_text = "PROBABLY NOT THIS AFTERNOON IN SAN DIEGO"
        token_ids = tokenizer.encode_as_ids(test_text)
        self.assertTrue(len(token_ids) > 0)
        decoded_text = tokenizer.decode(token_ids)
        self.assertIsInstance(decoded_text, str)

        print(f"Tokenizer loaded cleanly. Vocab size: {vocab_size}. Sample encoding: {token_ids[:5]}...")

        # Verify Teacher Model Modules
        modules = teacher_model.modules_dict
        self.assertIn("enc", modules)
        self.assertIn("output_emb", modules)
        self.assertIn("dec", modules)
        self.assertIn("seq_lin", modules)

        # Run dummy forward pass through encoder (B, T, Bins) -> (1, 100, 31)
        dummy_spec = torch.randn(1, 100, 31)
        with torch.no_grad():
            enc_out = teacher_model(dummy_spec)
        self.assertEqual(enc_out.shape[0], 1)
        self.assertEqual(enc_out.shape[2], 256)  # CRDNN encoder output dim

        print("StealthyIMU Teacher model and tokenizer loaded and verified successfully.")

    def test_03_test_dataset_loading_and_items(self):
        """Test loading all 3,070 test dataset items and validating shape & annotations."""
        logging_msg = "=== Step 3: Testing 3,070 Test Dataset Ingestion ==="
        print("\n" + logging_msg)

        dataset = load_test_dataset()
        total_items = len(dataset)
        self.assertEqual(total_items, 3070, f"Expected 3070 test dataset items, found {total_items}")

        # Inspect Sample 0
        sample_0 = dataset[0]
        self.assertIn("id", sample_0)
        self.assertIn("transcript", sample_0)
        self.assertIn("intent", sample_0)
        self.assertIn("imu_stft", sample_0)

        imu_stft = sample_0["imu_stft"]
        self.assertEqual(imu_stft.shape, (1, 128, 128))
        self.assertTrue(torch.isfinite(imu_stft).all())
        self.assertGreater(
            imu_stft.abs().sum().item(),
            0.0,
            "Loaded IMU STFT tensor must contain valid non-zero signal data."
        )

        transcript = sample_0["transcript"]
        intent = sample_0["intent"]
        self.assertTrue(len(transcript) > 0)
        self.assertNotEqual(intent, "")

        print(f"Test dataset loaded cleanly. Total items: {total_items}")
        print(f"Sample 0 ID: {sample_0['id']}")
        print(f"Sample 0 Intent: {intent}")
        print(f"Sample 0 Transcript: '{transcript[:40]}...'")
        print(f"Sample 0 IMU STFT shape: {imu_stft.shape}")

    def test_04_end_to_end_m1_pipeline_integration(self):
        """Test end-to-end integration: dataset -> generator reconstruction -> teacher encoder."""
        logging_msg = "=== Step 4: Testing M1 End-to-End Ingestion Integration ==="
        print("\n" + logging_msg)

        dataset = load_test_dataset(max_samples=5)
        generator = load_accear_generator()
        teacher_model, _ = load_slu_teacher_and_tokenizer()

        sample = dataset[0]
        imu_stft = sample["imu_stft"].unsqueeze(0)  # (1, 1, 128, 128)

        # 1. Reconstruct spectrogram via cGAN generator
        recon_spec = reconstruct_spectrogram(generator, imu_stft)
        self.assertEqual(recon_spec.shape, (1, 1, 128, 128))

        # 2. Pass reconstructed spectrogram directly through teacher encoder
        with torch.no_grad():
            encoder_feats = teacher_model(recon_spec)
        self.assertEqual(encoder_feats.shape[0], 1)
        self.assertEqual(encoder_feats.shape[2], 256)

        print("End-to-end M1 ingestion pipeline integration test PASSED.")

    def test_05_teacher_predict_raises_not_implemented(self):
        """Verify TeacherSLUModel.predict raises NotImplementedError (no facade implementations)."""
        logging_msg = "=== Step 5: Testing TeacherSLUModel.predict raises NotImplementedError ==="
        print("\n" + logging_msg)

        teacher_model, _ = load_slu_teacher_and_tokenizer()
        dummy_spec = torch.randn(1, 1, 128, 128)
        with self.assertRaises(NotImplementedError) as ctx:
            teacher_model.predict(dummy_spec)
        self.assertIn("Milestone M3", str(ctx.exception))
        print("TeacherSLUModel.predict correctly raises NotImplementedError.")

    def test_06_dataset_strict_mode_and_exceptions(self):
        """Verify strict mode path validation and exception handling in StealthyIMUTestDataset."""
        logging_msg = "=== Step 6: Testing Dataset Strict Mode & Path Exceptions ==="
        print("\n" + logging_msg)

        # Invalid metadata CSV path
        with self.assertRaises(FileNotFoundError):
            StealthyIMUTestDataset(metadata_csv="non_existent_path.csv")

        # Invalid dataset dir path
        with self.assertRaises(FileNotFoundError):
            StealthyIMUTestDataset(dataset_dir="non_existent_dir")

        print("Dataset strict mode path validation verified successfully.")

    def test_07_collate_fn_and_dataloader(self):
        """Verify stealthy_imu_collate_fn with PyTorch DataLoader (batch_size > 1)."""
        logging_msg = "=== Step 7: Testing stealthy_imu_collate_fn with DataLoader ==="
        print("\n" + logging_msg)

        dataset = load_test_dataset(max_samples=4)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=stealthy_imu_collate_fn,
        )

        for batch in dataloader:
            self.assertIn("imu_stft", batch)
            self.assertIn("semantics", batch)
            self.assertEqual(batch["imu_stft"].shape, (4, 1, 128, 128))
            self.assertEqual(len(batch["semantics"]), 4)
            self.assertIsInstance(batch["semantics"], list)
            self.assertEqual(len(batch["id"]), 4)
            self.assertEqual(batch["duration"].shape, (4,))

        print("stealthy_imu_collate_fn and DataLoader batching verified successfully.")

    def test_08_edge_cases_tensor_ranks_and_short_signals(self):
        """Verify tensor rank normalization and short signal padding."""
        logging_msg = "=== Step 8: Testing Rank Normalization & Short Signal STFT ==="
        print("\n" + logging_msg)

        generator = load_accear_generator()
        # 2D input tensor (128, 128) -> reconstruct_spectrogram
        imu_stft_2d = torch.rand(128, 128)
        recon_2d = reconstruct_spectrogram(generator, imu_stft_2d)
        self.assertEqual(recon_2d.shape, (1, 1, 128, 128))

        # 3D z tensor (1, 128, 128) -> reconstruct_spectrogram
        z_3d = torch.randn(1, 128, 128)
        recon_3d = reconstruct_spectrogram(generator, imu_stft_2d, z=z_3d)
        self.assertEqual(recon_3d.shape, (1, 1, 128, 128))

        # Short signal (< 254 samples) -> compute_imu_stft_spectrogram
        short_signal = np.random.randn(50).astype(np.float32)
        stft_short = compute_imu_stft_spectrogram(short_signal, fs=1000, n_fft=254, target_shape=(128, 128))
        self.assertEqual(stft_short.shape, (128, 128))
        self.assertTrue(np.isfinite(stft_short).all())

        # 2D tensor (100, 31) -> TeacherSLUModel._prepare_input_features
        teacher_model, _ = load_slu_teacher_and_tokenizer()
        spec_2d = torch.randn(100, 31)
        prep_out = teacher_model._prepare_input_features(spec_2d)
        self.assertEqual(prep_out.shape, (1, 100, 31))

        print("Rank normalization and short signal STFT padding verified successfully.")


if __name__ == "__main__":
    unittest.main()

