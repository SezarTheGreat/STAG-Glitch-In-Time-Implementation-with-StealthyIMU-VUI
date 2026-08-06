"""
Unit verification test suite for Milestone M3 - Model Decoding and NLP Post-Tuning.

Verifies:
1. Temperature scaling function execution, numerical stability, shape preservation, and T <= 0 validation.
2. Levenshtein edit distance, normalized string similarity, and intent parser.
3. VUI template rescoring across all 7 intent domains (alarm, media, call, calendar, weather, music, timer).
4. Phonetic & Levenshtein text error corrector on acoustic typos and misclassifications.
5. Integrated beam search decoding (beam_size=10..20, T=1.25) with SpeechBrain Teacher SLU model.
6. Unified AccEarSLUPipelineDecoder end-to-end integration across Stage 1, Stage 2, and Stage 3.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Day_14_Experiment_AccEar.src.data_and_models import (
    load_accear_generator,
    load_slu_teacher_and_tokenizer,
    load_test_dataset,
)
from Day_14_Experiment_AccEar.src.decoding_nlp import (
    apply_temperature_scaling,
    levenshtein_distance,
    string_similarity,
    extract_intent_from_decoded_text,
    VUITemplateRescorer,
    PhoneticErrorCorrector,
    IntegratedBeamSearchDecoder,
    AccEarSLUPipelineDecoder,
)


class TestM3Decoding(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"

    def test_01_temperature_scaling(self):
        """Test temperature scaling function on probability logits."""
        print("\n=== Step 1: Testing Temperature Scaling ===")

        logits = torch.tensor([[2.0, 1.0, 0.1, -1.0]], dtype=torch.float32)

        # 1. Standard temperature T = 1.0
        log_p_t1 = apply_temperature_scaling(logits, temperature=1.0)
        self.assertEqual(log_p_t1.shape, logits.shape)
        self.assertTrue(torch.all(torch.isfinite(log_p_t1)))
        self.assertTrue(torch.all(log_p_t1 <= 0.0))  # Log probabilities are <= 0

        # 2. Temperature scaling T = 1.25 (smoother distribution)
        log_p_t125 = apply_temperature_scaling(logits, temperature=1.25)
        self.assertEqual(log_p_t125.shape, logits.shape)
        self.assertTrue(torch.all(torch.isfinite(log_p_t125)))

        # Compare entropy / variance: T=1.25 should yield higher entropy (smoother probs)
        p_t1 = torch.exp(log_p_t1)
        p_t125 = torch.exp(log_p_t125)
        entropy_t1 = -torch.sum(p_t1 * log_p_t1)
        entropy_t125 = -torch.sum(p_t125 * log_p_t125)
        self.assertGreater(entropy_t125.item(), entropy_t1.item())

        # 3. Invalid temperature validation (T <= 0 raises ValueError)
        with self.assertRaises(ValueError):
            apply_temperature_scaling(logits, temperature=0.0)

        with self.assertRaises(ValueError):
            apply_temperature_scaling(logits, temperature=-1.0)

        print("Temperature Scaling passed distribution smoothing and validation tests.")

    def test_02_levenshtein_and_string_similarity(self):
        """Test Levenshtein distance, normalized similarity, and intent parser."""
        print("\n=== Step 2: Testing Levenshtein & String Similarity ===")

        # 1. Levenshtein distance
        self.assertEqual(levenshtein_distance("weather", "weather"), 0)
        self.assertEqual(levenshtein_distance("wether", "weather"), 1)
        self.assertEqual(levenshtein_distance("musik", "music"), 1)
        self.assertEqual(levenshtein_distance("sever", "seven"), 1)
        self.assertEqual(levenshtein_distance("alram", "alarm"), 2)

        # 2. String similarity score
        sim_exact = string_similarity("weather", "weather")
        self.assertEqual(sim_exact, 1.0)

        sim_close = string_similarity("wether", "weather")
        self.assertGreaterEqual(sim_close, 0.8)

        sim_diff = string_similarity("weather", "music")
        self.assertLess(sim_diff, 0.5)

        # 3. Intent extraction helper
        intent_1 = extract_intent_from_decoded_text("{'action': 'weather'| 'entities': []}")
        self.assertEqual(intent_1, "weather")

        intent_2 = extract_intent_from_decoded_text("{'action': 'alarm'}")
        self.assertEqual(intent_2, "alarm")

        intent_fallback = extract_intent_from_decoded_text("invalid json format")
        self.assertEqual(intent_fallback, "unknown")

        print("Levenshtein and String Similarity verification PASSED.")

    def test_03_vui_template_rescorer(self):
        """Test LM VUI Template Rescorer across all 7 intent domains."""
        print("\n=== Step 3: Testing VUI Template Rescorer ===")

        rescorer = VUITemplateRescorer()

        # 1. Template matching score across 7 domains
        domains = ["alarm", "media", "call", "calendar", "weather", "music", "timer"]
        test_phrases = {
            "alarm": "set alarm for seven am",
            "media": "pause music right now",
            "call": "call john on mobile",
            "calendar": "check schedule for today",
            "weather": "what is the weather forecast",
            "music": "play song by artist",
            "timer": "set timer for five minutes",
        }

        for domain in domains:
            phrase = test_phrases[domain]
            score, matched_domain = rescorer.compute_template_matching_score(phrase)
            self.assertGreater(score, 0.5, f"Expected high template match score for {phrase}")
            self.assertEqual(matched_domain, domain, f"Expected domain {domain}, got {matched_domain}")

        # 2. Rescore candidate hypotheses list
        candidates = [
            {"text": "set alram for sever am", "score": -0.50},
            {"text": "set alarm for seven am", "score": -0.55},  # Better LM match
            {"text": "random noise phrase", "score": -0.45},
        ]

        rescored = rescorer.rescore_hypotheses(candidates, alpha=0.4, beta=0.05)
        self.assertEqual(len(rescored), 3)

        # The canonical template candidate "set alarm for seven am" should win after LM rescoring
        top_cand = rescored[0]
        self.assertEqual(top_cand["text"], "set alarm for seven am")
        self.assertEqual(top_cand["inferred_intent"], "alarm")

        print("VUI Template Rescorer passed all 7 domain matching and hypothesis rescoring tests.")

    def test_04_phonetic_error_corrector(self):
        """Test Phonetic and Levenshtein Text Error Corrector."""
        print("\n=== Step 4: Testing Phonetic Error Corrector ===")

        corrector = PhoneticErrorCorrector()

        # 1. Word level corrections
        word_corr, changed = corrector.correct_word("wether")
        self.assertTrue(changed)
        self.assertEqual(word_corr, "weather")

        word_corr2, changed2 = corrector.correct_word("musik")
        self.assertTrue(changed2)
        self.assertEqual(word_corr2, "music")

        word_no_change, changed3 = corrector.correct_word("weather")
        self.assertFalse(changed3)
        self.assertEqual(word_no_change, "weather")

        # 2. Full text level corrections
        dirty_text = "set alram for sever am and check the wether"
        clean_text, Applied = corrector.correct_text(dirty_text)

        self.assertIn("alarm", clean_text)
        self.assertIn("seven", clean_text)
        self.assertIn("weather", clean_text)
        self.assertTrue(len(Applied) >= 3)

        print(f"Phonetic Error Corrector corrected '{dirty_text}' -> '{clean_text}' with {len(Applied)} fixes.")

    def test_05_integrated_beam_search_decoder(self):
        """Test IntegratedBeamSearchDecoder with SpeechBrain Teacher SLU model."""
        print("\n=== Step 5: Testing Integrated Beam Search Decoder ===")

        model_wrapper, tokenizer = load_slu_teacher_and_tokenizer()

        # Initialize beam search decoder with beam_size=15, T=1.25
        beam_decoder = IntegratedBeamSearchDecoder(beam_size=15, temperature=1.25)
        self.assertEqual(beam_decoder.beam_size, 15)
        self.assertEqual(beam_decoder.temperature, 1.25)

        # Synthetic spectrogram input (1, 1, 128, 128)
        spec_tensor = torch.rand(1, 1, 128, 128, dtype=torch.float32)
        candidates = beam_decoder.decode_spectrogram(model_wrapper, spec_tensor, top_k=5)

        self.assertTrue(len(candidates) > 0)
        self.assertLessEqual(len(candidates), 5)

        first_cand = candidates[0]
        self.assertIn("text", first_cand)
        self.assertIn("tokens", first_cand)
        self.assertIn("score", first_cand)
        self.assertIn("intent", first_cand)
        self.assertTrue(isinstance(first_cand["score"], float))

        print(f"Integrated Beam Search Decoder returned {len(candidates)} candidates. Top score: {first_cand['score']:.4f}")

    def test_06_accear_slu_pipeline_decoder_integration(self):
        """Test end-to-end AccEarSLUPipelineDecoder across Stage 1, Stage 2, and Stage 3."""
        print("\n=== Step 6: Testing Full AccEar SLU Pipeline Decoder Integration ===")

        generator = load_accear_generator()
        teacher_model, tokenizer = load_slu_teacher_and_tokenizer()
        dataset = load_test_dataset(max_samples=2)

        pipeline_decoder = AccEarSLUPipelineDecoder(
            generator=generator,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            beam_size=15,
            temperature=1.25,
            device=self.device,
        )

        sample = dataset[0]

        # Stage 1 Baseline
        res_s1 = pipeline_decoder.decode_sample(sample, stage=1)
        self.assertEqual(res_s1["stage"], 1)
        self.assertIn("predicted_text", res_s1)

        # Stage 2 DSP Filtered
        res_s2 = pipeline_decoder.decode_sample(sample, stage=2)
        self.assertEqual(res_s2["stage"], 2)
        self.assertIn("predicted_text", res_s2)

        # Stage 3 DSP + Beam Search (K=15, T=1.25) + LM Rescoring + Phonetic Corrector
        res_s3 = pipeline_decoder.decode_sample(sample, stage=3, beam_size=15, temperature=1.25)
        self.assertEqual(res_s3["stage"], 3)
        self.assertIn("predicted_text", res_s3)
        self.assertIn("predicted_intent", res_s3)
        self.assertEqual(res_s3["beam_size"], 15)
        self.assertEqual(res_s3["temperature"], 1.25)
        self.assertIn("corrections_applied", res_s3)
        self.assertIn("beam_candidates", res_s3)

        print("Full AccEar SLU Pipeline Decoder integration verification PASSED for Stages 1, 2, and 3.")


if __name__ == "__main__":
    unittest.main()
