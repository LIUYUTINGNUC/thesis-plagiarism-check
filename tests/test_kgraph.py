"""Tests for the knowledge graph module."""

import pytest

from thesischeck.core.semantic.kgraph import (
    DOMAIN_TERMS,
    DomainKnowledgeGraph,
    load_domain_terms,
)


@pytest.fixture
def kgraph():
    return DomainKnowledgeGraph()


class TestDomainKnowledgeGraph:
    def test_empty_graph_properties(self, kgraph):
        assert kgraph.node_count == 0
        assert kgraph.edge_count == 0

    def test_extract_entities_finds_domain_terms(self, kgraph):
        text = "This clinical trial studies diagnosis and treatment of chronic disease."
        entities = kgraph.extract_entities(text, DOMAIN_TERMS["medicine"])
        assert "clinical trial" in entities
        assert "diagnosis" in entities
        assert "treatment" in entities

    def test_build_from_text_creates_graph(self, kgraph):
        text = "Clinical trial for diagnosis. Treatment of chronic disease."
        kgraph.build_from_text(text, DOMAIN_TERMS["medicine"])
        assert kgraph.node_count > 0

    def test_build_relations_from_entities(self, kgraph):
        entities = ["clinical trial", "diagnosis", "treatment"]
        text = "The clinical trial focuses on diagnosis. Treatment follows diagnosis."
        relations = kgraph.build_relations(entities, text)
        assert len(relations) > 0
        for rel in relations:
            assert len(rel) == 3  # (subject, predicate, object)

    def test_match_graph_identical(self, kgraph):
        text = "Clinical trial for diagnosis and treatment of chronic disease."
        kgraph.build_from_text(text, DOMAIN_TERMS["medicine"])

        kgraph2 = DomainKnowledgeGraph()
        kgraph2.build_from_text(text, DOMAIN_TERMS["medicine"])

        result = kgraph.match_graph(kgraph2)
        assert result["overall_score"] > 0.5

    def test_match_graph_different(self, kgraph):
        text1 = "Clinical trial for diagnosis and treatment of chronic disease."
        text2 = "The weather is nice today and the sun is shining brightly."
        kgraph.build_from_text(text1, DOMAIN_TERMS["medicine"])
        kgraph2 = DomainKnowledgeGraph()
        kgraph2.build_from_text(text2, DOMAIN_TERMS["medicine"])
        result = kgraph.match_graph(kgraph2)
        assert 0.0 <= result["overall_score"] <= 1.0

    def test_idea_plagiarism_score(self, kgraph):
        original = "Clinical trial for diagnosis of heart disease. Treatment requires medication."
        suspect = "Clinical trial for diagnosis of cardiac disease. Treatment needs drugs."
        result = kgraph.idea_plagiarism_score(original, suspect, domain="medicine")
        assert "entity_overlap" in result
        assert "relation_similarity" in result
        assert "overall_idea_plagiarism_score" in result
        assert "verdict" in result
        assert 0.0 <= result["overall_idea_plagiarism_score"] <= 1.0

    def test_argument_flow_similarity(self, kgraph):
        text1 = "First, we analyze symptoms. Then we diagnose. Finally, we treat."
        text2 = "First, we examine data. Then we analyze. Finally, we conclude."
        kgraph.build_from_text(text1, DOMAIN_TERMS["medicine"])
        kgraph2 = DomainKnowledgeGraph()
        kgraph2.build_from_text(text2, DOMAIN_TERMS["medicine"])
        score = kgraph.argument_flow_similarity(kgraph2)
        assert 0.0 <= score <= 1.0


class TestLoadDomainTerms:
    def test_load_medicine_terms(self):
        terms = load_domain_terms("medicine")
        assert len(terms) > 0
        assert "clinical trial" in terms

    def test_load_cs_terms(self):
        terms = load_domain_terms("computer_science")
        assert len(terms) > 0

    def test_load_humanities_terms(self):
        terms = load_domain_terms("humanities")
        assert len(terms) > 0

    def test_unknown_discipline_returns_fallback(self):
        terms = load_domain_terms("nonexistent")
        assert len(terms) > 0  # should return general academic terms