import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.interpolate as interpolate
from unittest.mock import MagicMock

# Setup mock for k2 to prevent lazy import errors in SpeechBrain on Windows
sys.modules['k2'] = MagicMock()

# Setup SpeechBrain lazy import patching
import speechbrain.utils.importutils as iu
_old_getattr = iu.LazyModule.__getattr__
iu.LazyModule.__getattr__ = lambda self, attr: (_ for _ in ()).throw(
    AttributeError(attr)) if attr.startswith('__') else _old_getattr(self, attr)

import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

# Set intra-op threads to maximize CPU cores usage (18 cores)
torch.set_num_threads(18)

# Add paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stag_original')))
sys.path.append(os.path.dirname(__file__))

import train
from projects.stag_original.src.pipeline.dataset import get_stag_bifurcation
from projects.interpolation_experiments.pipeline_variants import extract_features_from_interp
from projects.interpolation_experiments.feature_boosting_exploration import apply_peaking_filter
from projects.stag_original.src.evaluation.metrics import calculate_wer, calculate_seer, calculate_ser

def prepare_peaking_data(hparams, upscaler_lgb, db_gain):
    data_folder = hparams["data_folder"]
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")
    
    tokenizer = hparams["tokenizer"]
    lgb_model = upscaler_lgb.model
    W = upscaler_lgb.W

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        uuid = os.path.basename(wav)[:-4]
        base_dir = os.path.dirname(wav)
        acc_path = os.path.join(hparams["data_folder"], base_dir, f"{uuid}.acc")
        gyro_path = os.path.join(hparams["data_folder"], base_dir, f"{uuid}.gyro")
        
        from scipy.io import wavfile
        wav_abs = os.path.join(hparams["data_folder"], wav)
        rate, wav_data = wavfile.read(wav_abs)
        duration = len(wav_data) / rate
        
        acc_odd, gyro_even, acc_even_target, t_even, t_odd = get_stag_bifurcation(acc_path, gyro_path, duration)
        
        # Baseline upscaling
        cs = interpolate.CubicSpline(t_odd, acc_odd, extrapolate=True)
        acc_interp = cs(t_even)
        feats = extract_features_from_interp(gyro_even, acc_interp, W=W)
        pred_even = lgb_model.predict(feats)
        
        reconstructed_z = np.zeros(len(acc_odd) + len(pred_even))
        reconstructed_z[0::2] = acc_odd
        reconstructed_z[1::2] = pred_even
        
        # Post-upscale Lowpass Butterworth (80Hz) to clean step noise
        sos = signal.butter(4, 80.0, 'lowpass', fs=400.0, output='sos')
        recon_lp = signal.sosfiltfilt(sos, reconstructed_z)
        
        # Apply peaking EQ centered at 150 Hz with a bandwidth of 140 Hz (covering 80 to 220 Hz range)
        boosted = apply_peaking_filter(recon_lp, f0=150.0, db_gain=db_gain, bw=140.0, fs=400.0)
        
        boosted = np.nan_to_num(boosted)
        
        # Resample to 500 Hz for downstream SLU
        t_source = np.arange(len(boosted)) * (1.0 / 400.0)
        t_target = np.arange(int(len(boosted) * 500.0 / 400.0)) * (1.0 / 500.0)
        f_resample = interpolate.interp1d(t_source, boosted, kind='cubic', fill_value="extrapolate")
        boosted_500 = f_resample(t_target)
        
        signal_tensor = torch.from_numpy(boosted_500).float().to('cpu')
        return signal_tensor

    sb.dataio.dataset.add_dynamic_item([test_data], audio_pipeline)

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
    
    return test_data, tokenizer

def run_peaking_eval(db_gain, upscaler_lgb, device="cpu"):
    hparams_file = "day_04_05_stag_recreation/hparams/paper_exact.yaml"
    overrides = {
        "seed": 1235,
        "data_folder": "common/data/StealthyIMU_dataset/",
        "csv_test": "day_04_05_stag_recreation/results/slu_baseline_paper/1235/test-type=direct.csv",
        "csv_train": "day_04_05_stag_recreation/results/slu_baseline_paper/1235/train-type=direct.csv",
        "csv_valid": "day_04_05_stag_recreation/results/slu_baseline_paper/1235/valid-type=direct.csv",
        "output_folder": "day_04_05_stag_recreation/results/slu_baseline_paper/1235",
        "tokenizer_file": "day_04_05_stag_recreation/pretrain/51_unigram.model"
    }
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    train.show_results_every = 500
    
    # Workaround: copy tokenizer
    tok_src = "day_04_05_stag_recreation/pretrain/51_unigram.model"
    tok_dst_dir = hparams["output_folder"] + "/save/SLURM_tokenizer"
    tok_dst = tok_dst_dir + "/tokenizer.ckpt"
    os.makedirs(tok_dst_dir, exist_ok=True)
    if not os.path.exists(tok_dst):
        import shutil
        shutil.copy2(tok_src, tok_dst)
        
    hparams["pretrainer"].collect_files()
    try:
        hparams["pretrainer"].load_collected(device=device)
    except TypeError:
        hparams["pretrainer"].load_collected()

    # Optimize decoding speed
    if "beam_searcher" in hparams:
        hparams["beam_searcher"].beam_size = 16

    test_set, tokenizer = prepare_peaking_data(hparams, upscaler_lgb, db_gain)

    slu_brain = train.SLU(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": device},
        checkpointer=hparams["checkpointer"],
    )
    slu_brain.tokenizer = tokenizer
    
    # Recover checkpoint (epoch 30)
    slu_brain.checkpointer.recover_if_possible()

    # Set wer_file path before evaluation
    slu_brain.hparams.wer_file = f"day_11_12_boosting_and_peaking/wer_peaking_{int(db_gain)}db.txt"
    if os.path.exists(slu_brain.hparams.wer_file):
        os.remove(slu_brain.hparams.wer_file)

    slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])
    
    wers = []
    seers = []
    sers = []
    
    for score in slu_brain.wer_metric.scores:
        ref_text = " ".join(score["ref_tokens"]).replace("; ", "").strip()
        hyp_text = " ".join(score["hyp_tokens"]).replace("; ", "").strip()
        
        wer = calculate_wer(ref_text, hyp_text)
        seer = calculate_seer(ref_text, hyp_text)
        ser = calculate_ser(ref_text, hyp_text)
        
        wers.append(wer)
        seers.append(seer)
        sers.append(ser)
        
    avg_wer = np.mean(wers) * 100.0 if wers else 0.0
    avg_ser = np.mean(sers) * 100.0 if sers else 0.0
    avg_seer = np.mean(seers) * 100.0 if seers else 0.0
    
    return avg_wer, avg_ser, avg_seer

def main():
    device = "cpu"
    upscaler_path = "models/stealthy_imu/upscaler.pkl"
    
    with open(upscaler_path, 'rb') as f:
        upscaler_lgb = pickle.load(f)

    for db in [6.0, 9.0]:
        wer, ser, seer = run_peaking_eval(db, upscaler_lgb, device)
        print(f"[RESULT] Peaking EQ (80-220Hz, +{db}dB) -> Downstream Teacher WER: {wer:.2f}%, SER: {ser:.2f}%, SEER: {seer:.2f}%")

if __name__ == "__main__":
    main()
