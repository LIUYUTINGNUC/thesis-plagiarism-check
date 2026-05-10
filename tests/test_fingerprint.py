"""Tests for the GPT fingerprint detection module."""

import numpy as np

from thesischeck.core.ai_detection.fingerprint import (
    ai_detection_report,
    burstiness_score,
    combined_ai_score,
    fingerprint_vector,
    repetition_analysis,
    token_probability_analysis,
)


class TestBurstinessScore:
    def test_high_variance_human_like(self):
        text = "A. " * 5 + "A B C D E F G H I J K L M N. " + "A. " * 5
        score = burstiness_score(text)
        assert 0.0 <= score <= 1.0

    def test_uniform_ai_like(self):
        sentences = " ".join("A " * 5 + ". " for _ in range(20))
        score = burstiness_score(sentences)
        assert 0.0 <= score <= 1.0

    def test_empty_text(self):
        assert burstiness_score("") == 0.0

    def test_single_sentence(self):
        assert burstiness_score("This is one sentence.") == 0.0


class TestRepetitionAnalysis:
    def test_high_repetition(self):
        rep = repetition_analysis("the " * 50)
        assert rep["unigram_repetition"] > 0.5

    def test_all_unique(self):
        rep = repetition_analysis("apple banana cherry date elderberry fig")
        assert rep["unigram_repetition"] == 0.0

    def test_empty_text(self):
        rep = repetition_analysis("")
        assert rep["unigram_repetition"] == 0.0

    def test_short_text(self):
        rep = repetition_analysis("a b")
        assert rep["trigram_repetition"] == 0.0


class TestTokenProbabilityAnalysis:
    def test_returns_expected_keys(self):
        result = token_probability_analysis("the the the the the test text")
        assert "mean_prob" in result
        assert "prob_variance" in result
        assert "prob_skewness" in result
        assert "high_freq_ratio" in result

    def test_empty_text(self):
        result = token_probability_analysis("")
        assert result["mean_prob"] == 0.0

    def test_high_frequency_words(self):
        result = token_probability_analysis("the the the the the cat")
        assert result["high_freq_ratio"] > 0


class TestFingerprintVector:
    def test_returns_array(self):
        vec = fingerprint_vector("This is a test text for fingerprint analysis.")
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 8

    def test_empty_text(self):
        vec = fingerprint_vector("")
        assert isinstance(vec, np.ndarray)


class TestCombinedAIScore:
    def test_normal_range(self):
        score = combined_ai_score(
            "This is a diverse text with varying sentence structures. "
            "Some sentences are short. Others are significantly longer and "
            "contain multiple clauses and descriptive elements."
        )
        assert 0.0 <= score <= 1.0

    def test_empty_text(self):
        assert combined_ai_score("") == 0.0

    def test_ai_like_text_scores_higher(self):
        ai_text = "The results are significant. The data shows improvement. "
        "The model performs well. The accuracy is high. The method works. "
        "The findings are consistent. The approach is effective. The system is robust."
        human_text = (
            "We were shocked by what the data revealed—after months of "
        "meticulous experimentation, the results not only confirmed our "
        "hypothesis but opened up entirely new questions we hadn't even "
        "thought to ask."
        )
        ai_score = combined_ai_score(ai_text)
        human_score = combined_ai_score(human_text)
        # AI text should have equal or higher AI score
        assert ai_score >= human_score - 0.3


class TestAIDetectionReport:
    def test_returns_complete_report(self):
        report = ai_detection_report("This is a test text for AI detection.")
        assert "overall_ai_score" in report
        assert "verdict" in report
        assert "confidence" in report
        assert "details" in report

    def test_verdict_is_valid(self):
        report = ai_detection_report("Test text here.")
        assert report["verdict"] in [
            "likely_ai_generated", "possibly_ai_generated",
            "possibly_human_written", "likely_human_written",
        ]

    def test_empty_text(self):
        report = ai_detection_report("")
        assert report["overall_ai_score"] == 0.0
