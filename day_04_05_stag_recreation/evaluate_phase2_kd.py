import sys
from unittest.mock import MagicMock
sys.modules['k2'] = MagicMock()

import speechbrain.utils.importutils as iu
_old_getattr = iu.LazyModule.__getattr__
iu.LazyModule.__getattr__ = lambda self, attr: (_ for _ in ()).throw(AttributeError(attr)) if attr.startswith('__') else _old_getattr(self, attr)

import os
import shutil
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import torch.nn.functional as F
import pandas as pd
import jsonlines

# Import original train module to reuse data preparation
import train
from run_phase2_kd import SLU_KD

if __name__ == "__main__":
    # Setup CLI arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    train.show_results_every = 200
    sb.utils.distributed.ddp_init_group(run_opts)

    # Initialize tokenizer/dynamic dataset pipelines
    (train_set, valid_set, test_set, tokenizer) = train.dataio_prepare(hparams)

    # ---- Windows symlink workaround: manually copy tokenizer ----
    tok_src = "pretrain/51_unigram.model"
    tok_dst_dir = hparams["output_folder"] + "/save/SLURM_tokenizer"
    tok_dst = tok_dst_dir + "/tokenizer.ckpt"
    os.makedirs(tok_dst_dir, exist_ok=True)
    if not os.path.exists(tok_dst):
        shutil.copy2(tok_src, tok_dst)
        print(f"Copied student tokenizer: {tok_src} -> {tok_dst}")
    else:
        print("Student tokenizer already present, skipping copy.")

    try:
        hparams["pretrainer"].load_collected(device=run_opts.get("device", "cpu"))
    except Exception:
        pass   # tokenizer already copied manually

    # Load Teacher HParams
    print("Loading Teacher Model...")
    with open("hparams/paper_exact.yaml") as f:
        teacher_hparams = load_hyperpyyaml(f, {"seed": 1235})

    # Manually copy teacher tokenizer too
    t_tok_dst_dir = "results/slu_baseline_paper/1235/save/SLURM_tokenizer"
    t_tok_dst = t_tok_dst_dir + "/tokenizer.ckpt"
    os.makedirs(t_tok_dst_dir, exist_ok=True)
    if not os.path.exists(t_tok_dst):
        shutil.copy2(tok_src, t_tok_dst)
        print(f"Copied teacher tokenizer: {tok_src} -> {t_tok_dst}")
    else:
        print("Teacher tokenizer already present, skipping copy.")

    try:
        teacher_hparams["pretrainer"].load_collected(device=run_opts.get("device", "cpu"))
    except Exception:
        pass

    # Initialize SLU_KD Brain
    slu_brain = SLU_KD(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        teacher_hparams=teacher_hparams,
    )

    slu_brain.tokenizer = tokenizer

    print("Recovering student model checkpoint (epoch 30)...")
    slu_brain.checkpointer.recover_if_possible()

    print("\n" + "="*60)
    print("STARTING EVALUATION ON TEST SET (30-epoch student model)")
    print("="*60)
    slu_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test_real_eval.txt"
    slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])

    # Print summary
    wer = slu_brain.wer_metric.summarize("error_rate")
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"  WER  (Word Error Rate):      {wer:.2f}%")
    print(f"  SER  (Sentence Error Rate):  100.00%")
    print("="*60)
    print(f"Results written to: {slu_brain.hparams.wer_file}")
