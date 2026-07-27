import numpy as np
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import module4_stag_engine as m4
from config import StagConfig

class TestModule4(unittest.TestCase):
    def setUp(self):
        self.config = StagConfig(target_axis=2)
        self.engine = m4.StagEngine(self.config)
        # 6 channels, 1000 samples (5 seconds at 200Hz)
        self.features = np.random.randn(6, 1000).astype(np.float32)
        
    def test_upsampling_dimensions(self):
        recon = self.engine.forward(self.features)
        
        # Expected shape is (1, 2 * 1000) = (1, 2000)
        self.assertEqual(recon.shape, (1, 2000))
        
    def test_cubic_spline_preservation(self):
        recon = self.engine.forward(self.features)
        
        # The even indices (0, 2, 4...) in the 400Hz signal should match the original 200Hz points exactly
        original_z = self.features[self.config.target_axis, :]
        recon_z_even = recon[0, 0::2]
        
        np.testing.assert_allclose(recon_z_even, original_z, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
