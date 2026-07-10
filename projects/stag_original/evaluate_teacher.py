"""
Evaluate the Teacher (Paper-Exact) Model on the StealthyIMU test set.

This script loads the 30-epoch teacher model checkpoint from
results/slu_baseline_paper/1235/save/ and evaluates it on the
real-environment test split using SpeechBrain's beam search decoder.

Usage:
    python evaluate_teacher.py hparams/paper_exact.yaml
"""
import sys
from unittest.mock import MagicMock
sys.modules['k2'] = MagicMock()

import speechbrain.utils.importutils as iu
_old_getattr = iu.LazyModule.__getattr__
iu.LazyModule.__getattr__ = lambda self, attr: (_ for _ in ()).throw(
    AttributeError(attr)) if attr.startswith('__') else _old_getattr(self, attr)

import os
import shutil
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import pandas as pd

# Import original train module (defines SLU brain + dataio_prepare)
import train

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    train.show_results_every = 100

    sb.utils.distributed.ddp_init_group(run_opts)

    # ----- Prepare data splits -----
    (train_set, valid_set, test_set, tokenizer) = train.dataio_prepare(hparams)

    # ----- Windows symlink workaround: manually copy tokenizer -----
    tok_src = "pretrain/51_unigram.model"
    tok_dst_dir = hparams["output_folder"] + "/save/SLURM_tokenizer"
    tok_dst = tok_dst_dir + "/tokenizer.ckpt"
    os.makedirs(tok_dst_dir, exist_ok=True)
    if not os.path.exists(tok_dst):
        shutil.copy2(tok_src, tok_dst)
        print(f"[INFO] Copied tokenizer: {tok_src} -> {tok_dst}")
    else:
        print("[INFO] Tokenizer already present.")

    try:
        hparams["pretrainer"].load_collected(device=run_opts.get("device", "cpu"))
    except Exception:
        pass  # tokenizer already copied manually

    # ----- Initialize SLU Brain (Teacher) -----
    slu_brain = train.SLU(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    slu_brain.tokenizer = tokenizer

    # ----- Recover teacher checkpoint (epoch 30) -----
    print("\n" + "=" * 64)
    print("  TEACHER MODEL EVALUATION -- StealthyIMU VUI")
    print("  Model: Paper-Exact (CRDNN 64->128 + 4-layer BiLSTM-256)")
    print("  Checkpoint: results/slu_baseline_paper/1235/save/CKPT+epoch_30")
    print("=" * 64)

    slu_brain.checkpointer.recover_if_possible()

    # ----- Run Evaluation -----
    print("\n[EVAL] Starting evaluation on the real-environment test set...")
    slu_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test_real_teacher_eval.txt"
    slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])

    # ----- Print Summary -----
    try:
        wer = slu_brain.wer_metric.summarize("error_rate")
        cer = slu_brain.cer_metric.summarize("error_rate")
        print("\n" + "=" * 64)
        print("  EVALUATION RESULTS -- TEACHER MODEL (30 epochs)")
        print("=" * 64)
        print(f"  Word Error Rate  (WER) : {wer:.2f}%")
        print(f"  Char Error Rate  (CER) : {cer:.2f}%")
        print("=" * 64)
    except Exception as e:
        print(f"\n[WARN] Could not summarize metrics: {e}")
        print("[INFO] Check the WER file for detailed results.")

    print(f"\n[INFO] Detailed WER file: {slu_brain.hparams.wer_file}")
    print("[DONE] Teacher model evaluation finished.")
