"""
Day 14 AccEar Experiment - Model Decoding & NLP Post-Tuning Module.

This module provides:
1. Temperature Scaling on softmax probability logits.
2. Levenshtein and Phonetic Similarity distance utilities.
3. LM / VUI Template Rescorer for the 7 VUI intent domains (alarm, media, call, calendar, weather, music, timer).
4. Phonetic & Levenshtein Text Error Corrector to repair speech-to-text / IMU reconstruction misclassifications.
5. Integrated Beam Search Decoder using SpeechBrain S2SRNNBeamSearcher (beam_size=10..20, T=1.25).
6. AccEarSLUPipelineDecoder unified decoder class for Stage 1 (Baseline), Stage 2 (DSP Filtered),
   and Stage 3 (DSP + Beam Search + Temp Scaling + LM Rescoring + Phonetic Error Corrector).
"""

import ast
import re
import logging
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import speechbrain.decoders as decoders

try:
    from Day_14_Experiment_AccEar.src.data_and_models import (
        UNetGenerator,
        TeacherSLUModel,
        reconstruct_spectrogram,
        reconstruct_day13_hybrid_features,
    )
    from Day_14_Experiment_AccEar.src.dsp_filtering import apply_dsp_pipeline
except ImportError:
    from src.data_and_models import (
        UNetGenerator,
        TeacherSLUModel,
        reconstruct_spectrogram,
        reconstruct_day13_hybrid_features,
    )
    from src.dsp_filtering import apply_dsp_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------------------------------
# 1. Temperature Scaling & Logits Processing
# ----------------------------------------------------------------------
def apply_temperature_scaling(logits: torch.Tensor, temperature: float = 1.25) -> torch.Tensor:
    """
    Applies Temperature Scaling (T > 0) to softmax probability logits.
    Log probabilities are returned: log_softmax(logits / T, dim=-1).

    Args:
        logits (torch.Tensor): Unscaled raw model logits tensor.
        temperature (float): Temperature scaling factor (T > 0). Default is 1.25.

    Returns:
        torch.Tensor: Temperature-scaled log-probabilities tensor.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature parameter T must be strictly positive (> 0), got {temperature}")

    scaled_logits = logits / temperature
    return F.log_softmax(scaled_logits, dim=-1)


# ----------------------------------------------------------------------
# 2. Levenshtein & String Similarity Metrics
# ----------------------------------------------------------------------
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes exact Levenshtein edit distance between strings s1 and s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def string_similarity(s1: str, s2: str) -> float:
    """
    Computes normalized string similarity score in [0.0, 1.0] based on Levenshtein distance.
    """
    s1_clean = s1.strip().lower()
    s2_clean = s2.strip().lower()
    max_len = max(len(s1_clean), len(s2_clean))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1_clean, s2_clean)
    return max(0.0, 1.0 - dist / float(max_len))


def extract_intent_from_decoded_text(decoded_text: str) -> str:
    """
    Extracts intent / action string from SpeechBrain decoded dictionary output
    or returns 'unknown' if not present.
    """
    if not decoded_text:
        return "unknown"

    cleaned = decoded_text.replace("|", ",")
    try:
        data = ast.literal_eval(cleaned)
        if isinstance(data, dict):
            return str(data.get("action", "unknown")).lower()
    except Exception:
        pass

    match = re.search(r"['\"]action['\"]\s*:\s*['\"]([^'\"]+)['\"]", decoded_text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    return "unknown"


# ----------------------------------------------------------------------
# 3. LM / VUI Template Rescorer
# ----------------------------------------------------------------------
class VUITemplateRescorer:
    """
    Language Model / Natural Language VUI Template Rescorer for the 7 primary intent domains:
    - alarm, media, call, calendar, weather, music, timer
    (as well as fallback dataset domains: air, navigation, reminder, stock, sun, time).
    """

    VUI_TEMPLATES: Dict[str, List[str]] = {
        "alarm": [
            "set alarm for",
            "cancel alarm",
            "turn off alarm",
            "set an alarm at",
            "wake me up at",
            "stop alarm",
            "add an alarm",
            "delete alarm",
            "turn on alarm",
            "alarm",
        ],
        "media": [
            "pause music",
            "play music",
            "next song",
            "previous track",
            "stop playback",
            "resume playback",
            "volume up",
            "volume down",
            "mute",
            "unmute",
            "media",
        ],
        "call": [
            "call",
            "dial",
            "answer call",
            "decline call",
            "hang up",
            "call back",
            "make a call",
            "phone call",
        ],
        "calendar": [
            "check schedule",
            "add event",
            "what is on my calendar",
            "schedule a meeting",
            "create event",
            "calendar",
            "events today",
        ],
        "weather": [
            "what is the weather",
            "is it going to rain",
            "temperature outside",
            "weather forecast",
            "how is the weather",
            "current temperature",
            "weather",
        ],
        "music": [
            "play song",
            "play album",
            "play playlist",
            "shuffle music",
            "what song is this",
            "music",
            "play track",
        ],
        "timer": [
            "set timer for",
            "cancel timer",
            "stop timer",
            "how much time left on timer",
            "pause timer",
            "start timer",
            "timer",
        ],
        # Extended dataset domains
        "air": ["air quality", "air index", "check air"],
        "navigation": ["navigate to", "turn right", "turn left", "directions to"],
        "reminder": ["remind me to", "set a reminder", "add reminder"],
        "stock": ["stock price", "share price", "market price"],
        "sun": ["sunrise time", "sunset time", "sun position"],
        "time": ["what time is it", "current time", "time in"],
    }

    def __init__(self, templates: Optional[Dict[str, List[str]]] = None):
        self.templates = templates if templates is not None else self.VUI_TEMPLATES

    def compute_template_matching_score(self, hypothesis_text: str) -> Tuple[float, str]:
        """
        Computes template matching score in [0.0, 1.0] and best matching intent domain
        for a given decoded candidate text hypothesis.
        """
        text_clean = hypothesis_text.strip().lower()
        if not text_clean:
            return 0.0, "unknown"

        best_score = 0.0
        best_intent = "unknown"

        for intent, tmpl_list in self.templates.items():
            for tmpl in tmpl_list:
                tmpl_clean = tmpl.strip().lower()

                # 1. Direct sub-string containment match
                if tmpl_clean in text_clean or text_clean in tmpl_clean:
                    score = 1.0
                else:
                    # 2. String similarity match
                    score = string_similarity(text_clean, tmpl_clean)

                if score > best_score:
                    best_score = score
                    best_intent = intent

        return best_score, best_intent

    def rescore_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        alpha: float = 0.4,
        beta: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Rescores candidate hypotheses combining acoustic score with LM VUI template matching:
            score_final = acoustic_score + alpha * lm_score + beta * len(text)

        Args:
            hypotheses (List[Dict[str, Any]]): List of candidate dicts with 'text' and 'score'.
            alpha (float): LM rescoring weight multiplier (default 0.4).
            beta (float): Length bonus weight (default 0.05).

        Returns:
            List[Dict[str, Any]]: Rescored candidates sorted by combined_score descending.
        """
        rescored = []
        for cand in hypotheses:
            cand_copy = dict(cand)
            ac_score = cand_copy.get("score", 0.0)
            text = cand_copy.get("text", "")

            lm_score, inferred_intent = self.compute_template_matching_score(text)
            len_bonus = float(len(text.split())) * beta if text else 0.0

            combined_score = ac_score + (alpha * lm_score) + len_bonus

            cand_copy["lm_score"] = lm_score
            cand_copy["inferred_intent"] = inferred_intent
            cand_copy["combined_score"] = combined_score
            rescored.append(cand_copy)

        rescored.sort(key=lambda x: x["combined_score"], reverse=True)
        return rescored


# ----------------------------------------------------------------------
# 4. Phonetic & Levenshtein Text Error Corrector
# ----------------------------------------------------------------------
class PhoneticErrorCorrector:
    """
    Lightweight text error corrector combining Levenshtein distance and phonetic dictionary matching
    to repair misclassifications in VUI command texts.
    """

    CANONICAL_VUI_WORDS = {
        "set", "alarm", "timer", "music", "play", "pause", "weather", "call", "dial",
        "schedule", "cancel", "stop", "volume", "meeting", "calendar", "forecast", "temperature",
        "seven", "eight", "nine", "ten", "one", "two", "three", "four", "five", "six", "zero",
        "minutes", "hours", "seconds", "today", "tomorrow", "morning", "afternoon", "evening", "night",
        "rain", "sun", "stock", "reminder", "navigation", "turn", "right", "left", "straight",
        "north", "south", "east", "west", "check", "add", "event", "create", "wake", "mute", "unmute"
    }

    COMMON_TYPOS_MAP = {
        "wether": "weather",
        "musik": "music",
        "muzic": "music",
        "sever": "seven",
        "alram": "alarm",
        "alaram": "alarm",
        "cal": "call",
        "timmer": "timer",
        "pos": "pause",
        "paws": "pause",
        "temprature": "temperature",
        "skedule": "schedule",
        "cancle": "cancel",
        "concel": "cancel",
        "volum": "volume",
        "minits": "minutes",
        "minuts": "minutes",
    }

    DEFAULT_STOPWORDS = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of", "and", "or", "am"
    }

    def __init__(
        self,
        canonical_words: Optional[set] = None,
        common_typos: Optional[Dict[str, str]] = None,
        stopwords: Optional[set] = None,
        min_token_len: int = 3,
    ):
        self.canonical_words = canonical_words if canonical_words is not None else self.CANONICAL_VUI_WORDS
        self.common_typos = common_typos if common_typos is not None else self.COMMON_TYPOS_MAP
        self.stopwords = stopwords if stopwords is not None else self.DEFAULT_STOPWORDS
        self.min_token_len = min_token_len

    def correct_word(self, word: str, max_edit_distance: int = 2) -> Tuple[str, bool]:
        """
        Corrects a single word if it matches a known typo or lies within max_edit_distance of a canonical VUI word.
        Prevents over-correction of short words (< min_token_len) or English stopwords.
        Returns (corrected_word, was_changed).
        """
        clean_word = word.strip().lower()
        if not clean_word:
            return word, False

        # Direct typo lookup
        if clean_word in self.common_typos:
            return self.common_typos[clean_word], True

        # Stopword whitelist or minimum token length constraint (>= 3 chars)
        if clean_word in self.stopwords or len(clean_word) < self.min_token_len:
            return word, False

        # Already canonical or valid
        if clean_word in self.canonical_words:
            return word, False

        # Search for closest canonical word by Levenshtein distance
        best_candidate = word
        min_dist = max_edit_distance + 1

        for target in self.canonical_words:
            dist = levenshtein_distance(clean_word, target)
            if dist < min_dist:
                min_dist = dist
                best_candidate = target

        if min_dist <= max_edit_distance and best_candidate != clean_word:
            return best_candidate, True

        return word, False

    def correct_text(self, text: str, max_edit_distance: int = 2) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Applies phonetic/Levenshtein error correction word-by-word across full input text string.

        Args:
            text (str): Input text string.
            max_edit_distance (int): Maximum edit distance threshold for word replacement (default 2).

        Returns:
            Tuple[str, List[Tuple[str, str]]]: Corrected text string and list of (original, corrected) pairs.
        """
        if not text:
            return text, []

        tokens = text.split(" ")
        corrected_tokens = []
        applied_corrections = []

        for token in tokens:
            # Preserve non-alphanumeric punctuation boundaries
            match = re.match(r"^(\W*)([a-zA-Z0-9]+)(\W*)$", token)
            if match:
                prefix, core_word, suffix = match.groups()
                corr_word, changed = self.correct_word(core_word, max_edit_distance=max_edit_distance)
                if changed:
                    applied_corrections.append((core_word, corr_word))
                    corrected_tokens.append(f"{prefix}{corr_word}{suffix}")
                else:
                    corrected_tokens.append(token)
            else:
                corr_word, changed = self.correct_word(token, max_edit_distance=max_edit_distance)
                if changed:
                    applied_corrections.append((token, corr_word))
                    corrected_tokens.append(corr_word)
                else:
                    corrected_tokens.append(token)

        corrected_text = " ".join(corrected_tokens)
        return corrected_text, applied_corrections


# ----------------------------------------------------------------------
# 5. Integrated Beam Search Decoder
# ----------------------------------------------------------------------
class IntegratedBeamSearchDecoder:
    """
    Beam Search Decoder integrated with SpeechBrain S2SRNNBeamSearcher.
    Supports configurable beam_size (10..20) and temperature scaling (T = 1.25).
    """

    def __init__(
        self,
        beam_size: int = 15,
        temperature: float = 1.25,
        bos_index: int = 0,
        eos_index: int = 0,
        min_decode_ratio: float = 0.0,
        max_decode_ratio: float = 10.0,
        eos_threshold: float = 1.5,
    ):
        if beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {beam_size}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.beam_size = beam_size
        self.temperature = temperature
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.min_decode_ratio = min_decode_ratio
        self.max_decode_ratio = max_decode_ratio
        self.eos_threshold = eos_threshold

    def decode_spectrogram(
        self,
        model_wrapper: TeacherSLUModel,
        spectrogram: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes beam search decoding on input spectrogram tensor using SpeechBrain S2SRNNBeamSearcher.

        Args:
            model_wrapper (TeacherSLUModel): Loaded StealthyIMU Teacher model wrapper.
            spectrogram (torch.Tensor): Spectrogram tensor (B, 1, 128, 128) or (B, T, 31).
            top_k (Optional[int]): Number of top candidates to return (defaults to self.beam_size).

        Returns:
            List[Dict[str, Any]]: List of candidate dicts:
                [{'text': str, 'tokens': List[int], 'score': float, 'intent': str}]
        """
        target_topk = top_k if top_k is not None else self.beam_size

        # 1. Run encoder forward pass
        model_wrapper.eval()
        with torch.no_grad():
            enc_out = model_wrapper(spectrogram)  # (B, T, 256)

        wav_lens = torch.ones(enc_out.size(0), device=enc_out.device)

        # 2. Build S2SRNNBeamSearcher with configured beam_size and temperature
        searcher = decoders.S2SRNNBeamSearcher(
            bos_index=self.bos_index,
            eos_index=self.eos_index,
            min_decode_ratio=self.min_decode_ratio,
            max_decode_ratio=self.max_decode_ratio,
            embedding=model_wrapper.modules_dict["output_emb"],
            decoder=model_wrapper.modules_dict["dec"],
            linear=model_wrapper.modules_dict["seq_lin"],
            beam_size=self.beam_size,
            topk=target_topk,
            temperature=self.temperature,
            eos_threshold=self.eos_threshold,
            return_topk=True,
        )

        with torch.no_grad():
            topk_hyps, topk_lengths, topk_scores, _ = searcher(enc_out, wav_lens)

        # 3. Extract candidates for the first item in batch
        candidates = []
        num_candidates = topk_hyps.size(1)
        max_len = topk_hyps.size(2)

        for k in range(num_candidates):
            rel_len = float(topk_lengths[0, k].item())
            tok_len = int(round(rel_len * max_len)) if rel_len <= 1.0 else int(rel_len)
            tok_seq = topk_hyps[0, k, :tok_len].tolist()
            cleaned_toks = [t for t in tok_seq if t != 0 or len(tok_seq) == 1]
            raw_text = model_wrapper.tokenizer.decode_ids(cleaned_toks if cleaned_toks else tok_seq)
            score = float(topk_scores[0, k].item())
            intent = extract_intent_from_decoded_text(raw_text)

            candidates.append({
                "text": raw_text,
                "tokens": tok_seq,
                "score": score,
                "intent": intent,
            })

        return candidates


# ----------------------------------------------------------------------
# 6. Unified AccEarSLUPipelineDecoder Class
# ----------------------------------------------------------------------
class AccEarSLUPipelineDecoder:
    """
    Unified AccEar SLU Pipeline Decoder supporting:
    - Stage 1: Baseline (AccEar Generator raw reconstruction + Greedy SLU decoding)
    - Stage 2: DSP Filtered (AccEar Generator + Adaptive Wiener & Savitzky-Golay DSP + Greedy SLU)
    - Stage 3: DSP + Beam Search (K=10..20) + Temp Scaling (T=1.25) + LM Rescoring + Phonetic Error Corrector.
    """

    def __init__(
        self,
        generator: UNetGenerator,
        teacher_model: TeacherSLUModel,
        tokenizer: Any,
        beam_size: int = 15,
        temperature: float = 1.25,
        device: str = "cpu"
    ):
        self.generator = generator
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.device = device

        self.beam_size = beam_size
        self.temperature = temperature

        self.beam_decoder = IntegratedBeamSearchDecoder(
            beam_size=beam_size,
            temperature=temperature,
        )
        self.lm_rescorer = VUITemplateRescorer()
        self.error_corrector = PhoneticErrorCorrector()

    def decode_sample(
        self,
        sample: Dict[str, Any],
        stage: int = 3,
        beam_size: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs end-to-end decoding for a single sample from StealthyIMUTestDataset.

        Args:
            sample (Dict[str, Any]): Sample dict containing 'imu_stft', 'transcript', 'intent', 'id'.
            stage (int): Experiment stage - 1 (Baseline), 2 (DSP Filtered), or 3 (DSP + Beam + Temp + LM + Error Correction).
            beam_size (Optional[int]): Override beam size if provided.
            temperature (Optional[float]): Override temperature scaling factor if provided.

        Returns:
            Dict[str, Any]: Detailed pipeline output result dict.
        """
        sample_id = sample.get("id", "unknown_sample")
        gt_transcript = sample.get("transcript", "")
        gt_intent = sample.get("intent", "unknown")
        imu_stft = sample["imu_stft"]

        # 1. Day 13 Hybrid Feature Reconstruction (InertiEAR VAD + Active-Speech Z-Score Scaling + STAG + AccSpec)
        acc_path = sample.get("accnpy_path", "")
        duration = sample.get("duration", 4.0)
        raw_recon_spec = reconstruct_day13_hybrid_features(acc_path, duration, device=self.device)

        eff_beam_size = beam_size if beam_size is not None else self.beam_size
        eff_temperature = temperature if temperature is not None else self.temperature

        if stage == 1:
            # Stage 1: Day 13 Hybrid Baseline (InertiEAR VAD + STAG + Greedy SLU)
            greedy_decoder = IntegratedBeamSearchDecoder(beam_size=1, temperature=1.0)
            candidates = greedy_decoder.decode_spectrogram(self.teacher_model, raw_recon_spec)
            best_cand = candidates[0] if candidates else {"text": "", "intent": "unknown", "score": 0.0}

            final_text = best_cand["text"]
            final_intent = best_cand["intent"]
            corrections = []
            rescored_hyps = candidates

        elif stage == 2:
            # Stage 2: Day 13 Hybrid + DSP Filtered (Wiener + Savitzky-Golay + Greedy SLU)
            filt_spec = apply_dsp_pipeline(raw_recon_spec)
            greedy_decoder = IntegratedBeamSearchDecoder(beam_size=1, temperature=1.0)
            candidates = greedy_decoder.decode_spectrogram(self.teacher_model, filt_spec)
            best_cand = candidates[0] if candidates else {"text": "", "intent": "unknown", "score": 0.0}

            final_text = best_cand["text"]
            final_intent = best_cand["intent"]
            corrections = []
            rescored_hyps = candidates

        elif stage == 3:
            # Stage 3: Day 13 Hybrid + DSP + Beam Search (K=15) + Temp Scaling (T=1.25) + LM Rescoring + Phonetic Error Corrector
            filt_spec = apply_dsp_pipeline(raw_recon_spec)

            beam_decoder = IntegratedBeamSearchDecoder(
                beam_size=eff_beam_size,
                temperature=eff_temperature,
            )
            raw_candidates = beam_decoder.decode_spectrogram(self.teacher_model, filt_spec)

            # LM / VUI Template Rescoring
            rescored_hyps = self.lm_rescorer.rescore_hypotheses(raw_candidates)
            top_hyp = rescored_hyps[0] if rescored_hyps else {"text": "", "inferred_intent": "unknown"}

            # Phonetic & Levenshtein Text Error Correction
            corrected_text, corrections = self.error_corrector.correct_text(top_hyp["text"])

            final_text = corrected_text
            final_intent = top_hyp.get("inferred_intent") or top_hyp.get("intent", "unknown")
            if final_intent == "unknown":
                _, final_intent = self.lm_rescorer.compute_template_matching_score(final_text)

        else:
            raise ValueError(f"Unsupported experiment stage: {stage}. Must be 1, 2, or 3.")

        return {
            "stage": stage,
            "sample_id": sample_id,
            "predicted_text": final_text,
            "predicted_intent": final_intent,
            "ground_truth_transcript": gt_transcript,
            "ground_truth_intent": gt_intent,
            "corrections_applied": corrections,
            "beam_candidates": rescored_hyps,
            "beam_size": eff_beam_size,
            "temperature": eff_temperature,
        }
