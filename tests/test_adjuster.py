"""Tests for the dynamic threshold adjuster."""

import pytest

from thesischeck.core.config.adjuster import (
    DynamicThresholdAdjuster,
    calculate_confidence_interval,
    get_threshold_recommendation,
)
from thesischeck.core.config.models import DisciplineConfig


@pytest.fixture
def base_config():
    return DisciplineConfig(name="test")


class TestDynamicThresholdAdjuster:
    def test_no_adjustment_with_defaults(self, base_config):
        adjuster = DynamicThresholdAdjuster(base_config)
        adjusted = adjuster.adjust_thresholds()
        assert adjusted.similarity.semantic_threshold == pytest.approx(
            base_config.similarity.semantic_threshold
        )

    def test_short_text_increases_threshold(self, base_config):
        adjuster = DynamicThresholdAdjuster(base_config)
        adjusted = adjuster.adjust_thresholds(text_length=100)
        assert adjusted.similarity.semantic_threshold > base_config.similarity.semantic_threshold

    def test_long_text_decreases_threshold(self, base_config):
        adjuster = DynamicThresholdAdjuster(base_config)
        adjusted = adjuster.adjust_thresholds(text_length=50000)
        assert adjusted.similarity.semantic_threshold < base_config.similarity.semantic_threshold

    def test_high_citation_increases_quota(self, base_config):
        adjuster = DynamicThresholdAdjuster(base_config)
        adjusted = adjuster.adjust_thresholds(citation_density=0.5)
        assert adjusted.citation.max_quotation_ratio > base_config.citation.max_quotation_ratio

    def test_many_references_reduces_novelty_requirement(self, base_config):
        adjuster = DynamicThresholdAdjuster(base_config)
        adjusted = adjuster.adjust_thresholds(reference_count=50)
        assert adjusted.citation.min_novel_finding_ratio < base_config.citation.min_novel_finding_ratio

    def test_adjustment_bounds(self, base_config):
        adjuster = DynamicThresholdAdjuster(base_config)
        # Very short text
        adjusted = adjuster.adjust_thresholds(text_length=10)
        assert adjusted.similarity.semantic_threshold <= 1.0


class TestConfidenceInterval:
    def test_empty_list(self):
        lower, upper = calculate_confidence_interval([])
        assert lower == 0.0 and upper == 0.0

    def test_single_value(self):
        lower, upper = calculate_confidence_interval([0.5])
        assert lower <= 0.5 <= upper

    def test_multiple_values(self):
        scores = [0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.2]
        lower, upper = calculate_confidence_interval(scores)
        assert 0.0 <= lower <= upper <= 1.0


class TestThresholdRecommendation:
    def test_returns_expected_keys(self, base_config):
        rec = get_threshold_recommendation(base_config, 5000, 15)
        assert "semantic_threshold" in rec
        assert "literal_threshold" in rec
        assert "max_quotation_ratio" in rec
        assert "adjustment_reasoning" in rec

    def test_short_text_reasoning(self, base_config):
        rec = get_threshold_recommendation(base_config, 100, 0)
        assert rec["adjustment_reasoning"]["length_factor"] == "short_text_increased_threshold"
