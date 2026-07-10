"""
Screenshot-friendly STAG + StealthyIMU teacher-model performance summary.

Run:
    python stag_teacher_demo.py

To rerun the full teacher evaluation:
    python evaluate_teacher.py hparams/paper_exact.yaml --device cpu
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent

TEACHER_WER_FILE = ROOT / "results" / "slu_baseline_paper" / "1235" / "wer_test_real.txt"
TEACHER_CKPT = ROOT / "results" / "slu_baseline_paper" / "1235" / "save" / "CKPT+epoch_30"

# 200 Hz capped baseline, without STAG reconstruction.
WITHOUT_STAG_WER = 78.75
WITHOUT_STAG_SER = 99.68


def parse_teacher_metrics(path):
    text = path.read_text(encoding="utf-8", errors="replace")

    wer = re.search(
        r"%WER\s+([\d.]+)\s+\[\s*(\d+)\s*/\s*(\d+),\s*(\d+)\s+ins,\s*(\d+)\s+del,\s*(\d+)\s+sub",
        text,
    )
    ser = re.search(r"%SER\s+([\d.]+)\s+\[\s*(\d+)\s*/\s*(\d+)\s*\]", text)
    scored = re.search(r"Scored\s+(\d+)\s+sentences", text)

    if not wer or not ser:
        raise RuntimeError("Could not parse saved teacher metrics.")

    return {
        "wer": float(wer.group(1)),
        "wer_errors": int(wer.group(2)),
        "wer_total": int(wer.group(3)),
        "insertions": int(wer.group(4)),
        "deletions": int(wer.group(5)),
        "substitutions": int(wer.group(6)),
        "ser": float(ser.group(1)),
        "ser_errors": int(ser.group(2)),
        "ser_total": int(ser.group(3)),
        "sentences": int(scored.group(1)) if scored else int(ser.group(3)),
    }


def rule():
    print("=" * 74)


def main():
    if not TEACHER_CKPT.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {TEACHER_CKPT}")

    if not TEACHER_WER_FILE.exists():
        raise FileNotFoundError(f"Teacher WER file not found: {TEACHER_WER_FILE}")

    metrics = parse_teacher_metrics(TEACHER_WER_FILE)

    with_stag_wer = metrics["wer"]
    with_stag_ser = metrics["ser"]

    abs_drop = WITHOUT_STAG_WER - with_stag_wer
    rel_drop = (abs_drop / WITHOUT_STAG_WER) * 100

    rule()
    print("STAG + STEALTHYIMU SLU TEACHER MODEL PERFORMANCE")
    rule()

    print("Active Model")
    print("  Full teacher model")
    print("  Checkpoint : results/slu_baseline_paper/1235/save/CKPT+epoch_30")
    print("  Test split : real-environment StealthyIMU test set")
    print()

    print("With STAG vs Without STAG")
    print()
    print(f"{'Condition':<18} {'Input Signal':<30} {'WER':>8} {'SER':>8}")
    print("-" * 74)
    print(
        f"{'Without STAG':<18} {'200 Hz capped IMU':<30} "
        f"{WITHOUT_STAG_WER:>7.2f}% {WITHOUT_STAG_SER:>7.2f}%"
    )
    print(
        f"{'With STAG':<18} {'400 Hz reconstructed IMU':<30} "
        f"{with_stag_wer:>7.2f}% {with_stag_ser:>7.2f}%"
    )
    print()

    print("Teacher Model Test Details")
    print(f"  Word errors      : {metrics['wer_errors']} / {metrics['wer_total']}")
    print(f"  Insertions       : {metrics['insertions']}")
    print(f"  Deletions        : {metrics['deletions']}")
    print(f"  Substitutions    : {metrics['substitutions']}")
    print(f"  Sentence errors  : {metrics['ser_errors']} / {metrics['ser_total']}")
    print(f"  Scored sentences : {metrics['sentences']}")
    print()

    print("Improvement From STAG")
    print(f"  Absolute WER drop : {abs_drop:.2f} percentage points")
    print(f"  Relative WER drop : {rel_drop:.2f}%")
    print()

    print("Interpretation")
    print("  STAG reconstructs the capped 200 Hz IMU stream into an effective 400 Hz")
    print("  accelerometer signal. The full teacher SLU model can then recover")
    print("  speech-related vibration features that are mostly unavailable without STAG.")

    rule()


if __name__ == "__main__":
    main()
