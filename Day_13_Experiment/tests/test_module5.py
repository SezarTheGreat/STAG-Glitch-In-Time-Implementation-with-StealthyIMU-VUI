import torch
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import module5_dual_inference as m5
from config import BranchAConfig, BranchBConfig

class TestModule5(unittest.TestCase):
    def setUp(self):
        # Mock reconstructed 400Hz signal batch: 2 samples, 1 channel, 2000 points
        self.x = torch.randn(2, 1, 2000)
        
    def test_branch_a_dimensions(self):
        config = BranchAConfig(vocab_size=100)
        model = m5.BranchA_Seq2Seq(config)
        
        # Test 10-token sequence generation
        logits = model(self.x, max_len=10)
        
        # Expected shape: (batch, seq_len, vocab_size)
        self.assertEqual(logits.shape, (2, 10, 100))
        
    def test_branch_b_dimensions(self):
        config = BranchBConfig(num_classes=10)
        model = m5.BranchB_DenseNet(config)
        
        logits = model(self.x)
        
        # Expected shape: (batch, num_classes)
        self.assertEqual(logits.shape, (2, 10))

if __name__ == '__main__':
    unittest.main()
