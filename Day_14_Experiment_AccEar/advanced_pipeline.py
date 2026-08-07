"""
Day 14 AccEar Experiment - Evaluation Pipeline Harness (advanced_pipeline.py).

This module implements the end-to-end evaluation harness for the AccEar IMU Speech Reconstruction
and SLU system across three experimental stages:
- Stage 1: Baseline (Raw AccEar Generator reconstruction + Greedy SLU decoding)
- Stage 2: Signal-Level DSP Filtered (AccEar Generator + Adaptive Wiener & Savitzky-Golay + Greedy SLU)
- Stage 3: Full Pipeline (AccEar Generator + DSP + Beam Search (K=15) + Temp Scaling (T=1.25) + LM Rescoring + Phonetic Corrector)

Metrics evaluated:
1. Word Error Rate (WER)
2. Sentence Error Rate (SER)
3. Intent Accuracy
4. Slot F1-Score (Precision, Recall, F1 across entity frame tuples)
5. Single Entity Error Rate (SEER)

Supports CLI execution via argparse for --dry-run (10 samples) and --full (3,070 samples).
Outputs JSON metrics report and human-readable text summary.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------
# Path Configurations & Imports
# ----------------------------------------------------------------------
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

try:
    from Day_14_Experiment_AccEar.src.data_and_models import (
        load_accear_generator,
        load_slu_teacher_and_tokenizer,
        load_test_dataset,
        stealthy_imu_collate_fn,
    )
    from Day_14_Experiment_AccEar.src.dsp_filtering import apply_dsp_pipeline
    from Day_14_Experiment_AccEar.src.decoding_nlp import (
        AccEarSLUPipelineDecoder,
        PhoneticErrorCorrector,
    )
    from day_04_05_stag_recreation.src.evaluation.metrics import (
        calculate_wer,
        calculate_ser,
        calculate_seer,
        parse_entity_frame,
    )
except ImportError:
    from src.data_and_models import (
        load_accear_generator,
        load_slu_teacher_and_tokenizer,
        load_test_dataset,
        stealthy_imu_collate_fn,
    )
    from src.dsp_filtering import apply_dsp_pipeline
    from src.decoding_nlp import (
        AccEarSLUPipelineDecoder,
        PhoneticErrorCorrector,
    )
    from day_04_05_stag_recreation.src.evaluation.metrics import (
        calculate_wer,
        calculate_ser,
        calculate_seer,
        parse_entity_frame,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------------------------------
# Helper Functions for Evaluation & Metrics
# ----------------------------------------------------------------------
def build_hypothesis_frame_str(pred_text: str, pred_intent: str) -> str:
    """
    Constructs a standardized semantic frame string representation for hypothesis evaluation.
    """
    pred_text_clean = pred_text.strip() if pred_text else ""
    pred_intent_clean = pred_intent.strip().lower() if pred_intent else "unknown"

    # If predicted_text is already formatted as dict string (e.g. SpeechBrain output)
    if pred_text_clean.startswith("{") and "action" in pred_text_clean:
        return pred_text_clean

    # Otherwise format a clean dict representation
    frame_dict = {"action": pred_intent_clean, "entities": []}

    # Extract any potential entities or words if present
    parsed_tuples = parse_entity_frame(pred_text_clean)
    entity_list = []
    for t_type, t_val in parsed_tuples:
        if t_type != "action" and t_type and t_val:
            entity_list.append({"type": t_type, "filler": t_val})

    if entity_list:
        frame_dict["entities"] = entity_list

    return str(frame_dict)


def compute_sample_metrics(
    gt_transcript: str,
    gt_intent: str,
    gt_semantics: Union[str, Dict[str, Any]],
    pred_text: str,
    pred_intent: str,
) -> Dict[str, Any]:
    """
    Computes single-sample metrics: WER, SER, Intent Accuracy, SEER, and entity tuples for Slot F1.
    """
    # 1. Format Reference and Hypothesis frame strings
    ref_frame_str = str(gt_semantics) if isinstance(gt_semantics, dict) else str(gt_semantics or "")
    hyp_frame_str = build_hypothesis_frame_str(pred_text, pred_intent)

    # 2. Word Error Rate (WER) against reference semantics frame tokens
    wer = calculate_wer(ref_frame_str, hyp_frame_str)

    # 3. Intent Accuracy
    gt_intent_clean = gt_intent.strip().lower() if gt_intent else "unknown"
    pred_intent_clean = pred_intent.strip().lower() if pred_intent else "unknown"
    intent_correct = 1.0 if pred_intent_clean == gt_intent_clean else 0.0

    # 4. Sentence Error Rate (SER) & Single Entity Error Rate (SEER)
    ser = calculate_ser(ref_frame_str, hyp_frame_str)
    seer = calculate_seer(ref_frame_str, hyp_frame_str)

    # 5. Extract entity tuples for Slot F1 calculation
    ref_tuples = parse_entity_frame(ref_frame_str)
    hyp_tuples = parse_entity_frame(hyp_frame_str)

    return {
        "wer": wer,
        "ser": ser,
        "intent_correct": intent_correct,
        "seer": seer,
        "ref_tuples": ref_tuples,
        "hyp_tuples": hyp_tuples,
    }


def compute_aggregate_metrics(sample_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes aggregate dataset metrics across all sample metrics.
    Includes WER, SER, Intent Accuracy, Slot F1 (Precision, Recall, F1), and SEER.
    """
    if not sample_metrics_list:
        return {
            "WER": 0.0,
            "SER": 0.0,
            "Intent_Accuracy": 0.0,
            "Slot_F1": 0.0,
            "Slot_Precision": 0.0,
            "Slot_Recall": 0.0,
            "SEER": 0.0,
        }

    total_samples = len(sample_metrics_list)
    avg_wer = float(sum(m["wer"] for m in sample_metrics_list) / total_samples)
    avg_ser = float(sum(m["ser"] for m in sample_metrics_list) / total_samples)
    avg_intent_acc = float(sum(m["intent_correct"] for m in sample_metrics_list) / total_samples)
    avg_seer = float(sum(m["seer"] for m in sample_metrics_list) / total_samples)

    # Calculate Slot F1-Score across all entity frame tuples
    total_tp = 0
    total_ref_tuples = 0
    total_hyp_tuples = 0

    for m in sample_metrics_list:
        ref_t = list(m["ref_tuples"])
        hyp_t = list(m["hyp_tuples"])

        total_ref_tuples += len(ref_t)
        total_hyp_tuples += len(hyp_t)

        # Count true positive tuple matches (multiset matching)
        ref_matched = [False] * len(ref_t)
        for ht in hyp_t:
            for idx, rt in enumerate(ref_t):
                if not ref_matched[idx] and ht == rt:
                    total_tp += 1
                    ref_matched[idx] = True
                    break

    precision = float(total_tp / total_hyp_tuples) if total_hyp_tuples > 0 else (1.0 if total_ref_tuples == 0 else 0.0)
    recall = float(total_tp / total_ref_tuples) if total_ref_tuples > 0 else 1.0
    if precision + recall > 0:
        f1_score = float(2.0 * precision * recall / (precision + recall))
    else:
        f1_score = 0.0

    return {
        "WER": round(avg_wer, 4),
        "SER": round(avg_ser, 4),
        "Intent_Accuracy": round(avg_intent_acc, 4),
        "Slot_F1": round(f1_score, 4),
        "Slot_Precision": round(precision, 4),
        "Slot_Recall": round(recall, 4),
        "SEER": round(avg_seer, 4),
    }


def compute_relative_change(baseline_val: float, new_val: float, lower_is_better: bool = True) -> float:
    """
    Computes percentage gain/reduction between baseline_val and new_val.
    Positive value indicates improvement.
    """
    if baseline_val == 0.0:
        return 0.0

    if lower_is_better:
        # Reduction in error rate (e.g. WER, SER, SEER)
        rel_change = (baseline_val - new_val) / baseline_val * 100.0
    else:
        # Gain in accuracy/F1 (e.g. Intent Accuracy, Slot F1)
        rel_change = (new_val - baseline_val) / baseline_val * 100.0

    return round(rel_change, 2)


def compute_comparative_gains(
    stage1: Dict[str, Any],
    stage2: Dict[str, Any],
    stage3: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Calculates relative comparative performance gains between pipeline stages:
    Stage 1 -> Stage 2 (DSP Filtering Gain)
    Stage 2 -> Stage 3 (NLP Post-Tuning Gain)
    Stage 1 -> Stage 3 (Overall Full Pipeline Gain)
    """
    return {
        "stage1_to_stage2": {
            "wer_reduction_pct": compute_relative_change(stage1["WER"], stage2["WER"], lower_is_better=True),
            "ser_reduction_pct": compute_relative_change(stage1["SER"], stage2["SER"], lower_is_better=True),
            "intent_acc_gain_pct": compute_relative_change(stage1["Intent_Accuracy"], stage2["Intent_Accuracy"], lower_is_better=False),
            "slot_f1_gain_pct": compute_relative_change(stage1["Slot_F1"], stage2["Slot_F1"], lower_is_better=False),
            "seer_reduction_pct": compute_relative_change(stage1["SEER"], stage2["SEER"], lower_is_better=True),
        },
        "stage2_to_stage3": {
            "wer_reduction_pct": compute_relative_change(stage2["WER"], stage3["WER"], lower_is_better=True),
            "ser_reduction_pct": compute_relative_change(stage2["SER"], stage3["SER"], lower_is_better=True),
            "intent_acc_gain_pct": compute_relative_change(stage2["Intent_Accuracy"], stage3["Intent_Accuracy"], lower_is_better=False),
            "slot_f1_gain_pct": compute_relative_change(stage2["Slot_F1"], stage3["Slot_F1"], lower_is_better=False),
            "seer_reduction_pct": compute_relative_change(stage2["SEER"], stage3["SEER"], lower_is_better=True),
        },
        "stage1_to_stage3_overall": {
            "wer_reduction_pct": compute_relative_change(stage1["WER"], stage3["WER"], lower_is_better=True),
            "ser_reduction_pct": compute_relative_change(stage1["SER"], stage3["SER"], lower_is_better=True),
            "intent_acc_gain_pct": compute_relative_change(stage1["Intent_Accuracy"], stage3["Intent_Accuracy"], lower_is_better=False),
            "slot_f1_gain_pct": compute_relative_change(stage1["Slot_F1"], stage3["Slot_F1"], lower_is_better=False),
            "seer_reduction_pct": compute_relative_change(stage1["SEER"], stage3["SEER"], lower_is_better=True),
        },
    }


def format_summary_text(
    metadata: Dict[str, Any],
    stage1: Dict[str, Any],
    stage2: Dict[str, Any],
    stage3: Dict[str, Any],
    gains: Dict[str, Dict[str, float]],
) -> str:
    """
    Formats human-readable text summary report of evaluation metrics and comparative gains.
    """
    summary = []
    summary.append("=" * 80)
    summary.append("Day 14 Experiment - Advanced Hybrid Evaluation Pipeline Summary Report")
    summary.append("=" * 80)
    summary.append(f"Timestamp        : {metadata.get('timestamp', 'N/A')}")
    summary.append(f"Total Samples    : {metadata.get('total_samples', 0)}")
    summary.append(f"Evaluation Device: {metadata.get('device', 'cpu')}")
    summary.append(f"Beam Size        : {metadata.get('beam_size', 15)}")
    summary.append(f"Temperature      : {metadata.get('temperature', 1.25)}")
    summary.append(f"Batch Size       : {metadata.get('batch_size', 8)}")
    summary.append(f"Execution Mode   : {metadata.get('mode', 'dry-run')}")
    summary.append("")

    summary.append("-" * 80)
    summary.append("STAGE PERFORMANCE SUMMARY")
    summary.append("-" * 80)

    stages = [
        ("Stage 1: Baseline (Day 13 Hybrid InertiEAR+STAG + Greedy SLU)", stage1),
        ("Stage 2: Signal-Level DSP Filtered (Wiener + SG Filter + Greedy SLU)", stage2),
        ("Stage 3: Full Advanced Pipeline (DSP + Beam Search + Temp + LM + Phonetic)", stage3),
    ]

    for name, s in stages:
        summary.append(name)
        summary.append(f"  - Word Error Rate (WER)        : {s['WER']:.4f} ({s['WER']*100:.2f}%)")
        summary.append(f"  - Sentence Error Rate (SER)    : {s['SER']:.4f} ({s['SER']*100:.2f}%)")
        summary.append(f"  - Intent Accuracy              : {s['Intent_Accuracy']:.4f} ({s['Intent_Accuracy']*100:.2f}%)")
        summary.append(f"  - Slot F1-Score                : {s['Slot_F1']:.4f} (Precision: {s['Slot_Precision']:.4f}, Recall: {s['Slot_Recall']:.4f})")
        summary.append(f"  - Single Entity Error Rate     : {s['SEER']:.4f} ({s['SEER']*100:.2f}%)")
        summary.append("")

    summary.append("-" * 80)
    summary.append("COMPARATIVE PERFORMANCE GAINS")
    summary.append("-" * 80)

    g12 = gains["stage1_to_stage2"]
    summary.append("Stage 1 -> Stage 2 (Signal-Level DSP Filtering Impact):")
    summary.append(f"  - WER Reduction                : {g12['wer_reduction_pct']:+.2f}%")
    summary.append(f"  - SER Reduction                : {g12['ser_reduction_pct']:+.2f}%")
    summary.append(f"  - Intent Accuracy Gain         : {g12['intent_acc_gain_pct']:+.2f}%")
    summary.append(f"  - Slot F1-Score Gain           : {g12['slot_f1_gain_pct']:+.2f}%")
    summary.append(f"  - SEER Reduction               : {g12['seer_reduction_pct']:+.2f}%")
    summary.append("")

    g23 = gains["stage2_to_stage3"]
    summary.append("Stage 2 -> Stage 3 (NLP Model Post-Tuning Impact):")
    summary.append(f"  - WER Reduction                : {g23['wer_reduction_pct']:+.2f}%")
    summary.append(f"  - SER Reduction                : {g23['ser_reduction_pct']:+.2f}%")
    summary.append(f"  - Intent Accuracy Gain         : {g23['intent_acc_gain_pct']:+.2f}%")
    summary.append(f"  - Slot F1-Score Gain           : {g23['slot_f1_gain_pct']:+.2f}%")
    summary.append(f"  - SEER Reduction               : {g23['seer_reduction_pct']:+.2f}%")
    summary.append("")

    g13 = gains["stage1_to_stage3_overall"]
    summary.append("Stage 1 -> Stage 3 (Full End-to-End Pipeline Cumulative Gain):")
    summary.append(f"  - WER Reduction                : {g13['wer_reduction_pct']:+.2f}%")
    summary.append(f"  - SER Reduction                : {g13['ser_reduction_pct']:+.2f}%")
    summary.append(f"  - Intent Accuracy Gain         : {g13['intent_acc_gain_pct']:+.2f}%")
    summary.append(f"  - Slot F1-Score Gain           : {g13['slot_f1_gain_pct']:+.2f}%")
    summary.append(f"  - SEER Reduction               : {g13['seer_reduction_pct']:+.2f}%")
    summary.append("=" * 80)

    return "\n".join(summary)


# ----------------------------------------------------------------------
# Evaluation Pipeline Runner
# ----------------------------------------------------------------------
def evaluate_dataset_samples(
    decoder: AccEarSLUPipelineDecoder,
    dataset: Any,
    stage: int = 3,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """
    Evaluates dataset samples sequentially or in safe memory batches for a specified stage.
    """
    metrics_list = []
    total_samples = len(dataset)

    logging.info(f"Evaluating Stage {stage} on {total_samples} samples (batch_size={batch_size})...")

    # DataLoader for batch memory management
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=stealthy_imu_collate_fn,
    )

    sample_idx = 0
    for batch in dataloader:
        batch_size_actual = len(batch["id"])
        for i in range(batch_size_actual):
            sample_idx += 1
            single_sample = {
                "id": batch["id"][i],
                "duration": batch["duration"][i].item() if isinstance(batch["duration"][i], torch.Tensor) else batch["duration"][i],
                "transcript": batch["transcript"][i],
                "intent": batch["intent"][i],
                "semantics": batch["semantics"][i],
                "imu_stft": batch["imu_stft"][i],
                "accnpy_path": batch["accnpy_path"][i],
            }

            # Run decoder step
            result = decoder.decode_sample(single_sample, stage=stage)

            # Compute sample metrics
            metrics = compute_sample_metrics(
                gt_transcript=single_sample["transcript"],
                gt_intent=single_sample["intent"],
                gt_semantics=single_sample["semantics"],
                pred_text=result["predicted_text"],
                pred_intent=result["predicted_intent"],
            )
            metrics_list.append(metrics)

            if sample_idx % 10 == 0 or sample_idx == total_samples:
                logging.info(f"Stage {stage} processed {sample_idx}/{total_samples} samples")

    return metrics_list


def run_evaluation_harness(
    dataset: Any,
    decoder: AccEarSLUPipelineDecoder,
    batch_size: int = 8,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes full evaluation harness across Stage 1, Stage 2, and Stage 3.
    """
    total_samples = len(dataset)
    timestamp_str = datetime.now(timezone.utc).isoformat()

    # Stage 1: Baseline
    logging.info("Starting Evaluation - Stage 1: Baseline...")
    s1_sample_metrics = evaluate_dataset_samples(decoder, dataset, stage=1, batch_size=batch_size)
    s1_agg = compute_aggregate_metrics(s1_sample_metrics)

    # Stage 2: DSP Filtered
    logging.info("Starting Evaluation - Stage 2: DSP Filtered...")
    s2_sample_metrics = evaluate_dataset_samples(decoder, dataset, stage=2, batch_size=batch_size)
    s2_agg = compute_aggregate_metrics(s2_sample_metrics)

    # Stage 3: Full Pipeline
    logging.info("Starting Evaluation - Stage 3: Full Pipeline...")
    s3_sample_metrics = evaluate_dataset_samples(decoder, dataset, stage=3, batch_size=batch_size)
    s3_agg = compute_aggregate_metrics(s3_sample_metrics)

    # Comparative gains
    gains = compute_comparative_gains(s1_agg, s2_agg, s3_agg)

    metadata = {
        "timestamp": timestamp_str,
        "total_samples": total_samples,
        "device": decoder.device,
        "beam_size": decoder.beam_size,
        "temperature": decoder.temperature,
        "batch_size": batch_size,
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    return {
        "metadata": metadata,
        "stage_1_baseline": s1_agg,
        "stage_2_dsp_filtered": s2_agg,
        "stage_3_full_pipeline": s3_agg,
        "comparative_gains": gains,
    }


# ----------------------------------------------------------------------
# CLI Interface & Main Function
# ----------------------------------------------------------------------
def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 14 AccEar Experiment - Evaluation Pipeline Harness"
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Process 10 test samples (or --num-samples) for quick pipeline verification.",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Process all 3,070 test sentences in safe memory batches.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional integer to limit the number of dataset samples evaluated.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for dataloader memory optimization (default: 8).",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="Day_14_Experiment_AccEar/outputs/advanced_metrics_report.json",
        help="Path to save output JSON metrics report.",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="Day_14_Experiment_AccEar/advanced_results_summary.txt",
        help="Path to save human-readable text summary report.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target execution device ('cuda' or 'cpu'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=15,
        help="Beam size K for Stage 3 Beam Search Decoding (default: 15).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.25,
        help="Temperature scaling factor T for Stage 3 decoding (default: 1.25).",
    )

    return parser.parse_args(args_list)


def main():
    args = parse_args()

    # Determine execution device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine sample count and mode
    if args.full:
        max_samples = args.num_samples  # None means all 3,070 samples
        mode_str = "full"
    elif args.dry_run:
        max_samples = args.num_samples if args.num_samples is not None else 10
        mode_str = "dry-run"
    else:
        # Default to dry-run behavior if neither --dry-run nor --full is passed
        max_samples = args.num_samples if args.num_samples is not None else 10
        mode_str = "dry-run"
        logging.info("Neither --dry-run nor --full specified. Defaulting to 10-sample dry-run mode.")

    logging.info(f"Initializing AccEar Evaluation Harness on device='{device}', mode='{mode_str}', max_samples={max_samples}")

    # Load Models and Tokenizer
    logging.info("Loading AccEar cGAN Generator...")
    generator = load_accear_generator(device=device)

    logging.info("Loading StealthyIMU SLU Teacher Model & Tokenizer...")
    teacher_model, tokenizer = load_slu_teacher_and_tokenizer(device=device)

    # Initialize Pipeline Decoder
    decoder = AccEarSLUPipelineDecoder(
        generator=generator,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        beam_size=args.beam_size,
        temperature=args.temperature,
        device=device,
    )

    # Load Test Dataset
    logging.info(f"Loading test dataset (max_samples={max_samples})...")
    dataset = load_test_dataset(max_samples=max_samples, strict=False)

    # Execute Evaluation Harness across Stage 1, Stage 2, and Stage 3
    results = run_evaluation_harness(
        dataset=dataset,
        decoder=decoder,
        batch_size=args.batch_size,
        metadata_extra={"mode": mode_str},
    )

    # Ensure output directories exist
    out_metrics_path = Path(args.output_metrics)
    out_summary_path = Path(args.output_summary)

    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON metrics report
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved JSON metrics report to {out_metrics_path.resolve()}")

    # Format and write text summary report
    summary_text = format_summary_text(
        metadata=results["metadata"],
        stage1=results["stage_1_baseline"],
        stage2=results["stage_2_dsp_filtered"],
        stage3=results["stage_3_full_pipeline"],
        gains=results["comparative_gains"],
    )

    with open(out_summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    logging.info(f"Saved text summary report to {out_summary_path.resolve()}")

    # Print summary to stdout
    print("\n" + summary_text + "\n")


if __name__ == "__main__":
    main()
