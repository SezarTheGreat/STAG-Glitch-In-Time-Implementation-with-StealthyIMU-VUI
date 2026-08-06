"""
Unit verification test suite for Milestone M4 - Evaluation Pipeline Harness.

Verifies:
1. CLI argument parsing (--dry-run, --full, --num-samples, --output-metrics, --output-summary, --device).
2. Metrics computation functions (WER, SER, Intent Accuracy, Slot F1-Score, SEER, relative comparative gains).
3. Dry-run evaluation harness execution on mock and dataset samples across Stage 1, Stage 2, and Stage 3.
4. JSON metrics report schema and text summary file format validation.
5. PhoneticErrorCorrector stopword whitelist and token length constraint protection against over-correction.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Day_14_Experiment_AccEar.advanced_pipeline import (
    parse_args,
    build_hypothesis_frame_str,
    compute_sample_metrics,
    compute_aggregate_metrics,
    compute_relative_change,
    compute_comparative_gains,
    format_summary_text,
    run_evaluation_harness,
)
from Day_14_Experiment_AccEar.src.decoding_nlp import (
    PhoneticErrorCorrector,
    AccEarSLUPipelineDecoder,
)
from Day_14_Experiment_AccEar.src.data_and_models import (
    load_accear_generator,
    load_slu_teacher_and_tokenizer,
    load_test_dataset,
)


class TestM4PipelineHarness(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"

    def test_01_cli_argument_parsing(self):
        """Test CLI argument parsing for --dry-run, --full, and custom options."""
        print("\n=== Step 1: Testing CLI Argument Parsing ===")

        # 1. Dry run default
        args_dry = parse_args(["--dry-run"])
        self.assertTrue(args_dry.dry_run)
        self.assertFalse(args_dry.full)
        self.assertEqual(args_dry.batch_size, 8)
        self.assertEqual(args_dry.beam_size, 15)
        self.assertEqual(args_dry.temperature, 1.25)

        # 2. Full mode with custom options
        args_full = parse_args([
            "--full",
            "--num-samples", "50",
            "--batch-size", "16",
            "--output-metrics", "custom_out.json",
            "--output-summary", "custom_summary.txt",
            "--device", "cpu",
            "--beam-size", "10",
            "--temperature", "1.5",
        ])
        self.assertTrue(args_full.full)
        self.assertFalse(args_full.dry_run)
        self.assertEqual(args_full.num_samples, 50)
        self.assertEqual(args_full.batch_size, 16)
        self.assertEqual(args_full.output_metrics, "custom_out.json")
        self.assertEqual(args_full.output_summary, "custom_summary.txt")
        self.assertEqual(args_full.device, "cpu")
        self.assertEqual(args_full.beam_size, 10)
        self.assertEqual(args_full.temperature, 1.5)

        print("CLI Argument Parsing tests PASSED.")

    def test_02_phonetic_error_corrector_whitelist_and_min_len(self):
        """Test PhoneticErrorCorrector stopword whitelist and token length protection."""
        print("\n=== Step 2: Testing PhoneticErrorCorrector Stopword & Length Constraints ===")

        corrector = PhoneticErrorCorrector(min_token_len=3)

        # 1. Stopwords should NOT be corrected
        stopwords = ["the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of", "and", "or", "am"]
        for sw in stopwords:
            corr, changed = corrector.correct_word(sw)
            self.assertFalse(changed, f"Stopword '{sw}' was incorrectly modified to '{corr}'")
            self.assertEqual(corr, sw)

        # 2. Tokens shorter than 3 chars (not in common typos) should NOT be corrected
        short_words = ["in", "at", "on", "to", "by", "if", "or", "so"]
        for sw in short_words:
            corr, changed = corrector.correct_word(sw)
            self.assertFalse(changed, f"Short word '{sw}' was incorrectly modified to '{corr}'")

        # 3. Known typos should still be corrected
        typo_corr, changed_typo = corrector.correct_word("wether")
        self.assertTrue(changed_typo)
        self.assertEqual(typo_corr, "weather")

        typo_corr2, changed_typo2 = corrector.correct_word("musik")
        self.assertTrue(changed_typo2)
        self.assertEqual(typo_corr2, "music")

        print("PhoneticErrorCorrector Whitelist and Length constraints PASSED.")

    def test_03_sample_and_aggregate_metrics_computation(self):
        """Test compute_sample_metrics, compute_aggregate_metrics, and comparative gains."""
        print("\n=== Step 3: Testing Metrics Computation & Comparative Gains ===")

        # 1. Perfect match sample
        sample_metrics_perfect = compute_sample_metrics(
            gt_transcript="what is the weather",
            gt_intent="weather",
            gt_semantics={'action': 'weather', 'entities': []},
            pred_text="what is the weather",
            pred_intent="weather",
        )
        self.assertEqual(sample_metrics_perfect["wer"], 0.0)
        self.assertEqual(sample_metrics_perfect["ser"], 0.0)
        self.assertEqual(sample_metrics_perfect["intent_correct"], 1.0)
        self.assertEqual(sample_metrics_perfect["seer"], 0.0)

        # 2. Mismatched sample
        sample_metrics_bad = compute_sample_metrics(
            gt_transcript="set alarm for seven am",
            gt_intent="alarm",
            gt_semantics={'action': 'alarm', 'entities': [{'type': 'time', 'filler': 'seven am'}]},
            pred_text="play music",
            pred_intent="music",
        )
        self.assertGreater(sample_metrics_bad["wer"], 0.0)
        self.assertEqual(sample_metrics_bad["intent_correct"], 0.0)
        self.assertEqual(sample_metrics_bad["ser"], 1.0)

        # 3. Aggregate metrics computation
        agg = compute_aggregate_metrics([sample_metrics_perfect, sample_metrics_bad])
        self.assertIn("WER", agg)
        self.assertIn("SER", agg)
        self.assertIn("Intent_Accuracy", agg)
        self.assertIn("Slot_F1", agg)
        self.assertIn("Slot_Precision", agg)
        self.assertIn("Slot_Recall", agg)
        self.assertIn("SEER", agg)

        self.assertEqual(agg["Intent_Accuracy"], 0.5)

        # 4. Comparative gains
        stage1 = {"WER": 0.50, "SER": 0.60, "Intent_Accuracy": 0.40, "Slot_F1": 0.40, "SEER": 0.50}
        stage2 = {"WER": 0.40, "SER": 0.50, "Intent_Accuracy": 0.50, "Slot_F1": 0.50, "SEER": 0.40}
        stage3 = {"WER": 0.20, "SER": 0.25, "Intent_Accuracy": 0.80, "Slot_F1": 0.75, "SEER": 0.20}

        gains = compute_comparative_gains(stage1, stage2, stage3)
        self.assertIn("stage1_to_stage2", gains)
        self.assertIn("stage2_to_stage3", gains)
        self.assertIn("stage1_to_stage3_overall", gains)

        # Stage 1 (0.50) -> Stage 3 (0.20) WER reduction = 60.0%
        self.assertEqual(gains["stage1_to_stage3_overall"]["wer_reduction_pct"], 60.0)

        print("Metrics Computation and Comparative Gains PASSED.")

    def test_04_dry_run_harness_execution_and_schema_validation(self):
        """Test run_evaluation_harness on dataset subset and validate JSON & TXT output schema."""
        print("\n=== Step 4: Testing Dry-Run Evaluation Harness & Output Schema ===")

        # Load models and 2-sample test dataset
        generator = load_accear_generator(device=self.device)
        teacher_model, tokenizer = load_slu_teacher_and_tokenizer(device=self.device)
        decoder = AccEarSLUPipelineDecoder(
            generator=generator,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            beam_size=5,
            temperature=1.25,
            device=self.device,
        )
        dataset = load_test_dataset(max_samples=2, strict=False)

        # Run evaluation harness
        results = run_evaluation_harness(
            dataset=dataset,
            decoder=decoder,
            batch_size=2,
            metadata_extra={"mode": "unit-test"},
        )

        # Validate JSON keys
        required_root_keys = ["metadata", "stage_1_baseline", "stage_2_dsp_filtered", "stage_3_full_pipeline", "comparative_gains"]
        for k in required_root_keys:
            self.assertIn(k, results)

        metadata = results["metadata"]
        self.assertEqual(metadata["total_samples"], 2)
        self.assertEqual(metadata["beam_size"], 5)
        self.assertEqual(metadata["temperature"], 1.25)
        self.assertEqual(metadata["mode"], "unit-test")

        metrics_keys = ["WER", "SER", "Intent_Accuracy", "Slot_F1", "SEER"]
        for stage_key in ["stage_1_baseline", "stage_2_dsp_filtered", "stage_3_full_pipeline"]:
            stage_dict = results[stage_key]
            for mk in metrics_keys:
                self.assertIn(mk, stage_dict)
                self.assertIsInstance(stage_dict[mk], (int, float))

        # Test writing JSON and TXT summary reports
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "test_report.json"
            txt_path = Path(tmp_dir) / "test_summary.txt"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f)

            summary_text = format_summary_text(
                metadata=results["metadata"],
                stage1=results["stage_1_baseline"],
                stage2=results["stage_2_dsp_filtered"],
                stage3=results["stage_3_full_pipeline"],
                gains=results["comparative_gains"],
            )

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(summary_text)

            self.assertTrue(json_path.exists())
            self.assertTrue(txt_path.exists())
            self.assertIn("Day 14 AccEar Experiment", summary_text)
            self.assertIn("STAGE PERFORMANCE SUMMARY", summary_text)

        print("Dry-Run Harness Execution and Schema Validation PASSED.")


if __name__ == "__main__":
    unittest.main()
