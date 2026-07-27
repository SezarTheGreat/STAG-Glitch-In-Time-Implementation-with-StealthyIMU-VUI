import numpy as np
import unittest
import sys
import os

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import module1_capture_denoise as m1
from config import PreprocessingConfig

class TestModule1(unittest.TestCase):
    def setUp(self):
        # Create synthetic 3-axis IMU data (Channels, Time) with DC bias
        self.accel = np.random.randn(3, 1000).astype(np.float32) + 5.0
        self.gyro = np.random.randn(3, 1000).astype(np.float32) - 2.0
        self.config = PreprocessingConfig()
        
    def test_dc_bias_removal(self):
        clean_accel = m1.remove_dc_bias(self.accel)
        means = np.mean(clean_accel, axis=1)
        np.testing.assert_allclose(means, np.zeros(3), atol=1e-5)
        
    def test_tensor_dimensions(self):
        accel_out, gyro_out = m1.process_raw_imu(self.accel, self.gyro, self.config)
        self.assertEqual(accel_out.shape, self.accel.shape, "Accel output dimension mismatch")
        self.assertEqual(gyro_out.shape, self.gyro.shape, "Gyro output dimension mismatch")
        
    def test_wiener_filter(self):
        filtered = m1.apply_wiener(self.accel, window_size=self.config.wiener_window_size)
        self.assertEqual(filtered.shape, self.accel.shape)
        # Verify it actually altered the signal slightly
        self.assertFalse(np.array_equal(filtered, self.accel))

    def test_1d_signal_compatibility(self):
        signal_1d = np.random.randn(1000) + 2.0
        out = m1.remove_dc_bias(signal_1d)
        self.assertAlmostEqual(np.mean(out), 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
