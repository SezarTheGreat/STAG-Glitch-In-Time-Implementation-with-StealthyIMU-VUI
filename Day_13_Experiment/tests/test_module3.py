import numpy as np
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import module3_normalization as m3
from config import NormalizationConfig

class TestModule3(unittest.TestCase):
    def setUp(self):
        self.config = NormalizationConfig(use_robust_scaling=False)
        # 6 channels (e.g. 3 accel + 3 gyro)
        self.features = np.random.randn(6, 1000) * 10.0 + 5.0
        self.speech_mask = np.zeros(1000, dtype=bool)
        self.speech_mask[400:600] = True
        
    def test_standard_scaling(self):
        norm = m3.apply_device_independent_scaling(self.features, self.speech_mask, self.config)
        
        # Verify the active region has ~0 mean and ~1 std
        active_norm = norm[:, self.speech_mask]
        means = np.mean(active_norm, axis=1)
        stds = np.std(active_norm, axis=1)
        
        np.testing.assert_allclose(means, np.zeros(6), atol=1e-5)
        np.testing.assert_allclose(stds, np.ones(6), atol=1e-5)
        
        # Ensure dimensions match
        self.assertEqual(norm.shape, self.features.shape)
        
    def test_robust_scaling(self):
        config_robust = NormalizationConfig(use_robust_scaling=True)
        norm = m3.apply_device_independent_scaling(self.features, self.speech_mask, config_robust)
        
        active_norm = norm[:, self.speech_mask]
        medians = np.median(active_norm, axis=1)
        np.testing.assert_allclose(medians, np.zeros(6), atol=1e-5)
        
    def test_empty_mask(self):
        empty_mask = np.zeros(1000, dtype=bool)
        norm = m3.apply_device_independent_scaling(self.features, empty_mask, self.config)
        # Should return un-normalized if no speech detected
        np.testing.assert_allclose(norm, self.features)

if __name__ == '__main__':
    unittest.main()
