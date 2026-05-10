"""Tests for the text preprocessor module."""

import pytest

from thesischeck.pipeline.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


class TestCleanText:
    def test_removes_extra_whitespace(self, preprocessor):
        result = preprocessor.clean_text("Hello    world.\n\n\nHow are you?")
        assert "Hello world." in result
        assert "How are you?" in result

    def test_handles_empty_text(self, preprocessor):
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text("   ") == ""

    def test_normalizes_newlines(self, preprocessor):
        result = preprocessor.clean_text("Line1\r\nLine2\rLine3")
        assert "Line1" in result
        assert "Line2" in result
        assert "Line3" in result


class TestSegmentSections:
    def test_detects_abstract(self, preprocessor):
        sections = preprocessor.segment_sections("Abstract\nThis is the abstract.")
        assert "abstract" in sections

    def test_detects_introduction(self, preprocessor):
        sections = preprocessor.segment_sections("1. Introduction\nThis is the intro.")
        assert "introduction" in sections

    def test_detects_methods(self, preprocessor):
        sections = preprocessor.segment_sections("Methods\nWe used a model.")
        assert "methods" in sections

    def test_detects_conclusion(self, preprocessor):
        sections = preprocessor.segment_sections("Conclusion\nWe conclude that.")
        assert "conclusion" in sections

    def test_preamble_before_first_section(self, preprocessor):
        sections = preprocessor.segment_sections(
            "Title of Paper\n\nAbstract\nContent here."
        )
        assert "preamble" in sections

    def test_empty_text(self, preprocessor):
        assert preprocessor.segment_sections("") == {}


class TestRemoveCitations:
    def test_removes_bracketed_citations(self, preprocessor):
        result = preprocessor.remove_citations("This is known [1, 2].")
        assert "[CITATION]" in result

    def test_removes_parenthetical_citations(self, preprocessor):
        result = preprocessor.remove_citations("As shown (Smith, 2020).")
        assert "[CITATION]" in result

    def test_handles_no_citations(self, preprocessor):
        text = "This is a normal sentence without citations."
        result = preprocessor.remove_citations(text)
        assert result == text


class TestExtractReferences:
    def test_extracts_references_section(self, preprocessor):
        text = "Some text.\nReferences\n[1] Author A. Title. Journal. 2020.\n[2] Author B. Title. 2021."
        refs = preprocessor.extract_references(text)
        assert len(refs) > 0

    def test_no_references_section(self, preprocessor):
        refs = preprocessor.extract_references("Just some text without references.")
        assert refs == []


class TestPrepareForAnalysis:
    def test_returns_expected_structure(self, preprocessor):
        result = preprocessor.prepare_for_analysis("This is a test paper. It has content.")
        assert "cleaned_text" in result
        assert "sections" in result
        assert "text_without_citations" in result
        assert "references" in result
        assert "metadata" in result

    def test_metadata_contains_word_count(self, preprocessor):
        result = preprocessor.prepare_for_analysis("Hello world.")
        assert result["metadata"]["word_count"] > 0

    def test_empty_input(self, preprocessor):
        result = preprocessor.prepare_for_analysis("")
        assert result["metadata"]["word_count"] == 0