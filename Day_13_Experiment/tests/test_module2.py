import numpy as np
import unittest
import sys
import os

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import module2_segmentation as m2
from config import SegmentationConfig, PreprocessingConfig

class TestModule2(unittest.TestCase):
    def setUp(self):
        self.prep_config = PreprocessingConfig(sampling_rate_in=200)
        self.seg_config = SegmentationConfig(
            otsu_bins=256,
            min_speech_duration_ms=50, # 10 samples
            energy_smooth_kernel=5,
            fill_gaps_ms=25 # 5 samples
        )
        
        # Create 1000 samples (5 seconds). 
        # Insert a high energy "speech" segment between 400 and 600.
        self.accel = np.random.randn(3, 1000).astype(np.float32) * 0.1
        self.gyro = np.random.randn(3, 1000).astype(np.float32) * 0.1
        
        # Inject speech
        self.accel[:, 400:600] += np.random.randn(3, 200) * 5.0
        self.gyro[:, 400:600] += np.random.randn(3, 200) * 5.0
        
    def test_energy_envelope(self):
        energy = m2.compute_energy_envelope(self.accel, self.gyro)
        self.assertEqual(energy.shape, (1000,))
        # Assert energy in speech segment is higher on average than noise segment
        self.assertGreater(np.mean(energy[400:600]), np.mean(energy[0:300]))
        
    def test_otsu_thresholding(self):
        energy = m2.compute_energy_envelope(self.accel, self.gyro)
        thresh = m2.apply_otsu_threshold(energy)
        self.assertGreater(thresh, np.min(energy))
        self.assertLess(thresh, np.max(energy))
        
    def test_segmentation_pipeline(self):
        mask = m2.segment_speech(self.accel, self.gyro, self.prep_config, self.seg_config)
        self.assertEqual(mask.shape, (1000,))
        self.assertEqual(mask.dtype, bool)
        
        # Verify speech is detected roughly around 400-600
        speech_detected_in_target = np.sum(mask[400:600])
        self.assertGreater(speech_detected_in_target, 100)
        
        speech_detected_in_noise = np.sum(mask[0:300])
        self.assertLess(speech_detected_in_noise, 50)
        
    def test_zero_variance_input(self):
        # Test completely flat signal
        flat_accel = np.zeros((3, 1000))
        flat_gyro = np.zeros((3, 1000))
        mask = m2.segment_speech(flat_accel, flat_gyro, self.prep_config, self.seg_config)
        self.assertFalse(np.any(mask)) # Should not trigger errors, likely returns all False

if __name__ == '__main__':
    unittest.main()
