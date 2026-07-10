import os
import sys
import pickle
import numpy as np
import scipy.interpolate as interpolate
from unittest.mock import MagicMock

# 1. Setup mock for k2 to prevent lazy import errors in SpeechBrain on Windows
sys.modules['k2'] = MagicMock()

# Setup SpeechBrain lazy import patching
import speechbrain.utils.importutils as iu
_old_getattr = iu.LazyModule.__getattr__
iu.LazyModule.__getattr__ = lambda self, attr: (_ for _ in ()).throw(
    AttributeError(attr)) if attr.startswith('__') else _old_getattr(self, attr)

import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stag_original')))

import train
from projects.combined_reconstruction.stacking import StackingUpscaler

def main():
    hparams_file = "projects/stag_original/hparams/paper_exact.yaml"
    overrides = {
        "seed": 1235,
        "data_folder": "common/data/StealthyIMU_dataset/",
        "csv_test": "projects/stag_original/results/slu_baseline_paper/1235/test-type=direct.csv",
        "output_folder": "projects/stag_original/results/slu_baseline_paper/1235",
        "tokenizer_file": "projects/stag_original/pretrain/51_unigram.model"
    }
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        
    # Load stacking upscaler
    with open("common/models/stacking_upscaler.pkl", "rb") as f:
        stacking_upscaler = pickle.load(f)
        
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test"], replacements={"data_root": hparams["data_folder"]},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")
    
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        uuid = os.path.basename(wav)[:-4]
        base_dir = os.path.dirname(wav)
        acc_path = os.path.join(hparams["data_folder"], base_dir, f"{uuid}.acc")
        gyro_path = os.path.join(hparams["data_folder"], base_dir, f"{uuid}.gyro")
        
        from projects.stag_original.src.pipeline.dataset import load_raw_sensor, get_stag_bifurcation
        t_gyro, _ = load_raw_sensor(gyro_path)
        duration = (t_gyro[-1] - t_gyro[0]) / 1000.0
        acc_odd, gyro_even, _, t_even, t_odd = get_stag_bifurcation(acc_path, gyro_path, duration)
        
        # Stacking model reconstruction
        reconstructed_z = stacking_upscaler.reconstruct_signal(acc_odd, gyro_even, t_odd, t_even)
        reconstructed_z = np.nan_to_num(reconstructed_z)
        
        # Resample to 500 Hz
        t_source = np.arange(len(reconstructed_z)) * (1.0 / 400.0)
        t_target = np.arange(int(len(reconstructed_z) * 500.0 / 400.0)) * (1.0 / 500.0)
        f_resample = interpolate.interp1d(t_source, reconstructed_z, kind='cubic', fill_value="extrapolate")
        reconstructed_z_500 = f_resample(t_target)
        
        return torch.from_numpy(reconstructed_z_500).float()
        
    sb.dataio.dataset.add_dynamic_item([test_data], audio_pipeline)
    
    # Text pipeline
    tokenizer = hparams["tokenizer"]
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides("semantics", "token_list", "tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item([test_data], text_pipeline)
    sb.dataio.dataset.set_output_keys([test_data], ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens"])
    
    test_data.data_ids = test_data.data_ids[:100]
    
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()
    
    slu_brain = train.SLU(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": "cpu"},
        checkpointer=hparams["checkpointer"],
    )
    slu_brain.tokenizer = tokenizer
    slu_brain.checkpointer.recover_if_possible()
    
    slu_brain.hparams.wer_file = "projects/interpolation_experiments/wer_stacking_test.txt"
    if os.path.exists(slu_brain.hparams.wer_file):
        os.remove(slu_brain.hparams.wer_file)
        
    slu_brain.evaluate(test_data, test_loader_kwargs=hparams["dataloader_opts"])
    
if __name__ == "__main__":
    main()
