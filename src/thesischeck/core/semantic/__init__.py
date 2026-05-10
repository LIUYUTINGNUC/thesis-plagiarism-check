"""Semantic analysis module: BERT encoding, similarity scoring, and knowledge graph construction."""

from thesischeck.core.semantic.encoder import SentenceEncoder
from thesischeck.core.semantic.kgraph import DomainKnowledgeGraph, load_domain_terms
from thesischeck.core.semantic.similarity import (
    SemanticSearcher,
    cosine_similarity,
    pairwise_similarity,
    semantic_distance,
)

__all__ = [
    "SentenceEncoder",
    "cosine_similarity",
    "pairwise_similarity",
    "semantic_distance",
    "SemanticSearcher",
    "DomainKnowledgeGraph",
    "load_domain_terms",
]

__version__ = "0.1.0"
