"""Tests for the statistical feature extraction module."""

import numpy as np

from thesischeck.core.ai_detection.features import (
    calculate_entropy,
    extract_all_features,
    feature_vector,
    function_word_ratio,
    readability_scores,
    sentence_length_stats,
    vocabulary_richness,
)


class TestEntropy:
    def test_entropy_empty_text(self):
        assert calculate_entropy("") == 0.0
        assert calculate_entropy("   ") == 0.0

    def test_entropy_single_char(self):
        assert calculate_entropy("aaaaaa") == 0.0

    def test_entropy_varied_text(self):
        e = calculate_entropy("abc123!@#中文")
        assert e > 0.0


class TestSentenceLengthStats:
    def test_stats_empty_text(self):
        stats = sentence_length_stats("")
        assert stats["mean"] == 0.0

    def test_stats_basic(self):
        stats = sentence_length_stats("A B C. D E F G.")
        assert stats["mean"] > 0
        assert stats["std"] >= 0

    def test_stats_multiple_sentences(self):
        stats = sentence_length_stats("A. B. C. D. E.")
        assert stats["min"] <= stats["mean"] <= stats["max"]


class TestVocabularyRichness:
    def test_empty_text(self):
        assert vocabulary_richness("") == 0.0

    def test_all_unique(self):
        assert vocabulary_richness("apple banana cherry") == 1.0

    def test_all_repeated(self):
        assert vocabulary_richness("the the the the") == 0.25


class TestFunctionWordRatio:
    def test_empty_text(self):
        assert function_word_ratio("") == 0.0

    def test_english_function_words(self):
        ratio = function_word_ratio("the cat and the dog")
        assert 0.4 <= ratio <= 0.6

    def test_chinese_function_words(self):
        ratio = function_word_ratio("这是一个测试的句子")
        assert ratio >= 0.0


class TestReadabilityScores:
    def test_empty_text(self):
        scores = readability_scores("")
        assert scores["flesch_reading_ease"] == 0.0

    def test_easy_text(self):
        scores = readability_scores("The cat sat on the mat. It was happy.")
        assert scores["flesch_reading_ease"] >= 0

    def test_difficult_text(self):
        scores = readability_scores(
            "The philosophical underpinnings of phenomenological methodology "
            "require comprehensive epistemological consideration."
        )
        assert scores["fk_grade_level"] > 5


class TestExtractAllFeatures:
    def test_returns_all_keys(self):
        features = extract_all_features("A short test sentence.")
        assert "entropy" in features
        assert "vocabulary_richness" in features
        assert "sentence_length_mean" in features
        assert "flesch_reading_ease" in features


class TestFeatureVector:
    def test_returns_numpy_array(self):
        vec = feature_vector("Test text here.")
        assert isinstance(vec, np.ndarray)

    def test_normalized_range(self):
        vec = feature_vector("A" * 100)
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
