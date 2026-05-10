"""Tests for the pipeline orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from thesischeck.core.config.models import CheckResult
from thesischeck.pipeline.orchestrator import PlagiarismChecker, SemanticResult


@pytest.fixture
def checker():
    """Create a checker with mocked SentenceEncoder to avoid model downloads."""
    with patch("thesischeck.pipeline.orchestrator.SentenceEncoder") as mock_enc_cls:
        mock_enc = MagicMock()
        mock_enc.embedding_dim = 384
        mock_enc.model_name = "all-MiniLM-L6-v2"
        mock_enc._split_sentences.return_value = ["sentence one", "sentence two"]
        mock_enc.encode.return_value = __import__("numpy").random.rand(2, 384).astype("float32")
        mock_enc_cls.return_value = mock_enc

        yield PlagiarismChecker(discipline="default")


class TestPlagiarismChecker:
    def test_init_with_default_config(self):
        with patch("thesischeck.pipeline.orchestrator.SentenceEncoder"):
            checker = PlagiarismChecker()
        assert checker.discipline_name == "default"

    def test_init_with_custom_discipline(self):
        with patch("thesischeck.pipeline.orchestrator.SentenceEncoder"):
            checker = PlagiarismChecker(discipline="medicine")
        assert checker.discipline_name == "medicine"

    def test_literal_similarity_identical(self, checker):
        text = "The quick brown fox jumps over the lazy dog."
        score = checker._literal_similarity(text, text)
        assert score > 0.9

    def test_literal_similarity_different(self, checker):
        score = checker._literal_similarity(
            "This is the original text.",
            "Completely different content here."
        )
        assert score < 0.5

    def test_generate_verdict_highly_similar(self, checker):
        verdict = checker._generate_verdict(0.95, checker.config)
        assert verdict == "highly_similar"

    def test_generate_verdict_distinct(self, checker):
        verdict = checker._generate_verdict(0.1, checker.config)
        assert verdict == "distinct"

    @patch("thesischeck.pipeline.orchestrator.DomainKnowledgeGraph")
    def test_check_returns_check_result(self, mock_kg, checker):
        mock_kg_instance = MagicMock()
        mock_kg_instance.idea_plagiarism_score.return_value = {
            "overall_idea_plagiarism_score": 0.3,
            "entity_overlap": 0.3,
            "relation_similarity": 0.3,
            "argument_flow_score": 0.3,
            "verdict": "moderately_similar",
        }
        mock_kg.return_value = mock_kg_instance

        result = checker.check(
            "Original paper text here. It contains research content.",
            "Suspect paper text here. It might be similar.",
        )
        assert isinstance(result, CheckResult)
        assert result.discipline == "default"
        assert 0.0 <= result.overall_score <= 1.0

    def test_ai_check_returns_result(self, checker):
        result = checker._ai_check("This is a test text for AI detection.")
        assert hasattr(result, "ai_score")
        assert hasattr(result, "ai_verdict")
        assert result.ai_verdict in [
            "likely_ai_generated", "possibly_ai_generated",
            "possibly_human_written", "likely_human_written",
            "not_checked",
        ]


class TestIntegration:
    @patch("thesischeck.pipeline.orchestrator.DomainKnowledgeGraph")
    def test_semantic_check_integration(self, mock_kg, checker):
        mock_kg_instance = MagicMock()
        mock_kg_instance.idea_plagiarism_score.return_value = {
            "overall_idea_plagiarism_score": 0.2,
        }
        mock_kg.return_value = mock_kg_instance

        result = checker._semantic_check(
            "Original research text for testing.",
            "Suspect research text for testing.",
        )
        assert hasattr(result, "similarity_score")
        assert hasattr(result, "kgraph_score")
        assert hasattr(result, "literal_score")

    def test_discipline_config_loaded_correctly(self):
        with patch("thesischeck.pipeline.orchestrator.SentenceEncoder"):
            checker = PlagiarismChecker(discipline="cs")
        assert checker.config.name == "cs"
        assert checker.config.similarity.semantic_threshold == 0.7