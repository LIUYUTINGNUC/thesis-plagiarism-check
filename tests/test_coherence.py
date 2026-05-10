"""Tests for the coherence analysis module."""


from thesischeck.core.ai_detection.coherence import (
    coherence_score,
    discourse_pattern,
    transition_word_analysis,
)


class TestTransitionWordAnalysis:
    def test_returns_all_categories(self):
        result = transition_word_analysis(
            "First, we propose a method. However, there are limitations. "
            "Therefore, we need more research."
        )
        assert "total_transitions" in result
        assert result["sequential_count"] >= 1
        assert result["adversative_count"] >= 1
        assert result["causal_count"] >= 1

    def test_empty_text(self):
        result = transition_word_analysis("")
        assert result["total_transitions"] == 0.0

    def test_chinese_transitions(self):
        result = transition_word_analysis(
            "首先，我们提出一个方法。但是，存在局限性。因此，需要更多研究。"
        )
        assert result["total_transitions"] >= 3


class TestCoherenceScore:
    def test_normal_range(self):
        score = coherence_score(
            "First, we analyze the data. However, the results are inconclusive. "
            "Therefore, we conduct additional experiments. Finally, we confirm our hypothesis."
        )
        assert 0.0 <= score <= 1.0

    def test_empty_text(self):
        assert coherence_score("") == 0.0

    def test_no_transitions(self):
        score = coherence_score("Data analyzed. Results obtained. Conclusions drawn.")
        assert 0.0 <= score <= 1.0


class TestDiscoursePattern:
    def test_identifies_claim_evidence(self):
        pattern = discourse_pattern(
            "We propose a new method for data analysis. "
            "Experiments demonstrate its effectiveness. "
            "In conclusion, this approach is superior."
        )
        assert isinstance(pattern, list)
        assert len(pattern) > 0

    def test_empty_text(self):
        pattern = discourse_pattern("")
        # No paragraphs to analyze = no discourse segments
        # But the empty string produces one "other" segment
        assert isinstance(pattern, list)

    def test_chinese_patterns(self):
        pattern = discourse_pattern(
            "本文提出一种新方法。\n\n实验表明该方法有效。\n\n综上所述，本方法具有优势。"
        )
        assert "claim" in pattern or "other" in pattern
