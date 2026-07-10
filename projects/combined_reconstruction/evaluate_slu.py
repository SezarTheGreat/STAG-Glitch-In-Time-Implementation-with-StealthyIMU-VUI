import os
import sys
import re
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

# Add paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stag_original')))

import train

def parse_wer_file(filepath):
    """
    Parses WER, CER, and SER error metrics from the generated wer file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"WER file not found: {filepath}")
        
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    wer_match = re.search(
        r"%WER\s+([\d.]+)\s+\[\s*(\d+)\s*/\s*(\d+),\s*(\d+)\s+ins,\s*(\d+)\s+del,\s*(\d+)\s+sub",
        text,
    )
    ser_match = re.search(r"%SER\s+([\d.]+)\s+\[\s*(\d+)\s*/\s*(\d+)\s*\]", text)
    
    if not wer_match or not ser_match:
        raise RuntimeError(f"Could not parse metrics from {filepath}")
        
    return {
        "wer": float(wer_match.group(1)),
        "wer_errors": int(wer_match.group(2)),
        "wer_total": int(wer_match.group(3)),
        "ser": float(ser_match.group(1)),
        "ser_errors": int(ser_match.group(2)),
        "ser_total": int(ser_match.group(3)),
    }

def run_evaluation_with_upscaling(use_stacking_upscaler, output_wer_file, device="cpu"):
    """
    Runs SpeechBrain SLU Teacher evaluation loop on a sliced subset of the test dataset.
    """
    hparams_file = "projects/stag_original/hparams/paper_exact.yaml"
    
    # Define path overrides to match segregated directory structure
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

    hparams["use_stacking_upscaler"] = use_stacking_upscaler

    train.show_results_every = 100
    
    # 2. Windows symlink workaround: manually copy tokenizer
    tok_src = "projects/stag_original/pretrain/51_unigram.model"
    tok_dst_dir = hparams["output_folder"] + "/save/SLURM_tokenizer"
    tok_dst = tok_dst_dir + "/tokenizer.ckpt"
    os.makedirs(tok_dst_dir, exist_ok=True)
    if not os.path.exists(tok_dst):
        import shutil
        shutil.copy2(tok_src, tok_dst)
        print(f"[INFO] Copied tokenizer: {tok_src} -> {tok_dst}")

    # Download and pretrain/collect tokenizer and models
    hparams["pretrainer"].collect_files()
    try:
        hparams["pretrainer"].load_collected(device=device)
    except TypeError:
        hparams["pretrainer"].load_collected()

    # 3. Prepare data splits
    (train_set, valid_set, test_set, tokenizer) = train.dataio_prepare(hparams)
    
    # Subset the test set to the first 100 samples to keep CPU runtime down
    test_set.data_ids = test_set.data_ids[:100]
    print(f"[INFO] Running evaluation on subset of {len(test_set.data_ids)} sentences (use_stacking_upscaler={use_stacking_upscaler})...")

    # 4. Initialize SLU Brain (Teacher)
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

    # Set custom WER output file path
    slu_brain.hparams.wer_file = output_wer_file
    if os.path.exists(output_wer_file):
        os.remove(output_wer_file)

    # 5. Run evaluate
    slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])
    print(f"[SUCCESS] Evaluation output saved to {output_wer_file}")
    
    wer = slu_brain.wer_metric.summarize("error_rate")
    cer = slu_brain.cer_metric.summarize("error_rate")
    
    # Count sentence errors (SER)
    ser_errors = sum(1 for s in slu_brain.wer_metric.scores if s.get('num_edits', 0) > 0)
    total_sentences = len(slu_brain.wer_metric.scores)
    ser = (ser_errors / max(1, total_sentences)) * 100.0
    
    return {
        "wer": wer,
        "cer": cer,
        "ser": ser,
        "total": total_sentences
    }

def main():
    device = "cpu"
    original_wer_file = "projects/stag_original/results/slu_baseline_paper/1235/wer_test_subset_original.txt"
    stacking_wer_file = "projects/stag_original/results/slu_baseline_paper/1235/wer_test_subset_stacking.txt"
    
    print("\n" + "=" * 60)
    print("  SLU EVALUATION LOOP -- ORIGINAL STAG RECONSTRUCTION")
    print("=" * 60)
    original_metrics = run_evaluation_with_upscaling(
        use_stacking_upscaler=False,
        output_wer_file=original_wer_file,
        device=device
    )
    
    print("\n" + "=" * 60)
    print("  SLU EVALUATION LOOP -- TEACHER-LED STACKING CHAMPION RECONSTRUCTION")
    print("=" * 60)
    stacking_metrics = run_evaluation_with_upscaling(
        use_stacking_upscaler=True,
        output_wer_file=stacking_wer_file,
        device=device
    )
    
    # Print comparison
    print("\n" + "=" * 80)
    print("STAG + STEALTHYIMU SLU EVALUATION RESULTS -- COMPARATIVE SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<36} {'WER':>10} {'CER':>10} {'SER':>10} {'Sentences':>10}")
    print("-" * 80)
    # Original paper restricted baseline (pre-computed reference)
    print(f"{'Without STAG (200Hz Capped)':<36} {'78.75%':>10} {'Not Rep':>10} {'99.68%':>10} {'3070':>10}")
    # Original STAG reconstruction (evaluated on our subset)
    print(f"{'Original STAG (Cubic + LGBM)':<36} {original_metrics['wer']:>9.2f}% {original_metrics['cer']:>9.2f}% {original_metrics['ser']:>9.2f}% {original_metrics['total']:>10}")
    # Our Stacking Champion reconstruction (evaluated on our subset)
    print(f"{'Stacking Champion (New Stack)':<36} {stacking_metrics['wer']:>9.2f}% {stacking_metrics['cer']:>9.2f}% {stacking_metrics['ser']:>9.2f}% {stacking_metrics['total']:>10}")
    print("=" * 80)
    
    # Save results to a text file for review integration
    with open("projects/combined_reconstruction/slu_evaluation_results.txt", "w") as f:
        f.write(f"Original STAG - WER: {original_metrics['wer']:.2f}%, CER: {original_metrics['cer']:.2f}%, SER: {original_metrics['ser']:.2f}%\n")
        f.write(f"Stacking Champion - WER: {stacking_metrics['wer']:.2f}%, CER: {stacking_metrics['cer']:.2f}%, SER: {stacking_metrics['ser']:.2f}%\n")

if __name__ == "__main__":
    main()
