import os
import sys
import pickle
from unittest.mock import MagicMock

sys.modules['k2'] = MagicMock()
import speechbrain.utils.importutils as iu
_old_getattr = iu.LazyModule.__getattr__
iu.LazyModule.__getattr__ = lambda self, attr: (_ for _ in ()).throw(
    AttributeError(attr)) if attr.startswith('__') else _old_getattr(self, attr)

import torch
import speechbrain as sb

# Add paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stag_original')))
sys.path.append(os.path.dirname(__file__))

from hyperpyyaml import load_hyperpyyaml

# Set up logging to print INFO level logs
import logging
logging.basicConfig(level=logging.INFO)

hparams_file = "projects/stag_original/hparams/paper_exact.yaml"
overrides = {
    "seed": 1235,
    "data_folder": "common/data/StealthyIMU_dataset/",
    "csv_test": "projects/stag_original/results/slu_baseline_paper/1235/test-type=direct.csv",
    "csv_train": "projects/stag_original/results/slu_baseline_paper/1235/train-type=direct.csv",
    "csv_valid": "projects/stag_original/results/slu_baseline_paper/1235/valid-type=direct.csv",
    "output_folder": "projects/stag_original/results/slu_baseline_paper/1235",
    "tokenizer_file": "projects/stag_original/pretrain/51_unigram.model"
}

with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

print("Checkpoints directory configured as:", hparams["checkpointer"].checkpoints_dir)
print("Checking path existence:", os.path.exists(hparams["checkpointer"].checkpoints_dir))

# Try to recover checkpoint
try:
    recovered = hparams["checkpointer"].recover_if_possible()
    print("Recovery result:", recovered)
except Exception as e:
    print("Recovery failed with exception:", e)
