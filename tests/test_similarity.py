"""Tests for semantic similarity functions and the FAISS-based SemanticSearcher."""

import numpy as np
import pytest

from thesischeck.core.semantic.similarity import (
    SemanticSearcher,
    cosine_similarity,
    pairwise_similarity,
    semantic_distance,
)


# ======================================================================
# cosine_similarity
# ======================================================================


class TestCosineSimilarity:
    """Unit tests for :func:`cosine_similarity`."""

    def test_cosine_similarity_identical(self):
        """Identical vectors → similarity 1.0."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v1, v2) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors → similarity 0.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self):
        """Opposite (anti-parallel) vectors → similarity -1.0."""
        v1 = np.array([2.0, 3.0])
        v2 = np.array([-2.0, -3.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self):
        """Zero vector → similarity 0.0 (to avoid division by zero)."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v1, v2) == 0.0

        # Both zero
        assert cosine_similarity(v1, v1) == 0.0

    def test_cosine_similarity_arbitrary(self):
        """Arbitrary vectors produce the expected cosine value."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, 0.5])
        # dot = 0.5, |v1| = 1, |v2| = sqrt(0.5) ≈ 0.7071
        # similarity = 0.5 / 0.7071 ≈ 0.7071
        expected = 0.5 / np.sqrt(0.5)
        assert cosine_similarity(v1, v2) == pytest.approx(expected)


# ======================================================================
# pairwise_similarity
# ======================================================================


class TestPairwiseSimilarity:
    """Unit tests for :func:`pairwise_similarity`."""

    def test_pairwise_similarity_shape(self):
        """(n, d) input → (n, n) output."""
        matrix = np.random.rand(5, 10)
        result = pairwise_similarity(matrix)
        assert result.shape == (5, 5)

    def test_pairwise_similarity_diagonal(self):
        """Diagonal entries are all 1.0 (self-similarity)."""
        matrix = np.random.rand(4, 8)
        result = pairwise_similarity(matrix)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-6)

    def test_pairwise_similarity_symmetric(self):
        """Output matrix is symmetric."""
        matrix = np.random.rand(6, 12)
        result = pairwise_similarity(matrix)
        assert np.allclose(result, result.T, atol=1e-6)

    def test_pairwise_similarity_single_vector(self):
        """Single vector → 1x1 matrix with value 1.0."""
        matrix = np.array([[3.0, 4.0]])
        result = pairwise_similarity(matrix)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0)

    def test_pairwise_similarity_with_zero_rows(self):
        """Rows of all zeros are handled without error (similarity = 0)."""
        matrix = np.array([[0.0, 0.0], [1.0, 2.0]])
        result = pairwise_similarity(matrix)
        assert result.shape == (2, 2)
        # Row 0 is zero → its self-similarity is 0 (not 1)
        assert result[0, 0] == 0.0
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# ======================================================================
# semantic_distance
# ======================================================================


class TestSemanticDistance:
    """Unit tests for :func:`semantic_distance`."""

    def test_semantic_distance_identical(self):
        """Identical vectors → distance 0.0."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])
        assert semantic_distance(v1, v2) == pytest.approx(0.0)

    def test_semantic_distance_orthogonal(self):
        """Orthogonal vectors → distance 1.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert semantic_distance(v1, v2) == pytest.approx(1.0)

    def test_semantic_distance_relation(self):
        """distance = 1 - similarity holds for arbitrary vectors."""
        v1 = np.array([2.0, -1.0, 3.0])
        v2 = np.array([-1.0, 4.0, 0.5])
        sim = cosine_similarity(v1, v2)
        dist = semantic_distance(v1, v2)
        assert dist == pytest.approx(1.0 - sim)


# ======================================================================
# SemanticSearcher
# ======================================================================


class TestSemanticSearcher:
    """Integration tests for :class:`SemanticSearcher` (requires FAISS)."""

    def test_searcher_build_and_search(self):
        """Build index → search returns the closest vectors."""
        searcher = SemanticSearcher()
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]], dtype=np.float32)
        searcher.build_index(vectors)

        # Query vector identical to the third vector
        query = np.array([0.707, 0.707], dtype=np.float32)
        results = searcher.search(query, k=3)

        assert len(results) == 3
        # Top result should have index 2 (the third vector) and score ~1.0
        top_idx, top_score = results[0]
        assert top_idx == 2
        assert top_score == pytest.approx(1.0, abs=1e-3)

    def test_searcher_empty_index(self):
        """Searching an empty (or unbuilt) index returns an empty list."""
        searcher = SemanticSearcher()
        query = np.array([1.0, 2.0], dtype=np.float32)
        assert searcher.search(query) == []

    def test_searcher_k_clamping(self):
        """k > index size returns all available results (no crash)."""
        searcher = SemanticSearcher()
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        searcher.build_index(vectors)

        query = np.array([1.0, 0.0], dtype=np.float32)
        results = searcher.search(query, k=99)  # k > ntotal

        assert len(results) == 2  # clamped to index size

    def test_searcher_index_size_property(self):
        """``index_size`` reports the correct count."""
        searcher = SemanticSearcher()
        assert searcher.index_size == 0

        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        searcher.build_index(vectors)
        assert searcher.index_size == 3

    def test_searcher_add_vectors(self):
        """``add`` appends to an existing index."""
        searcher = SemanticSearcher()
        searcher.build_index(np.array([[1.0, 0.0]], dtype=np.float32))
        assert searcher.index_size == 1

        searcher.add(np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32))
        assert searcher.index_size == 3

    def test_searcher_search_1d_query(self):
        """1-D query vector is handled without error."""
        searcher = SemanticSearcher()
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        searcher.build_index(vectors)

        query = np.array([1.0, 0.0])  # 1-D, not explicitly float32
        results = searcher.search(query, k=2)
        assert len(results) == 2

    def test_searcher_results_sorted(self):
        """Results are returned in descending order of similarity."""
        searcher = SemanticSearcher()
        vectors = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float32
        )
        searcher.build_index(vectors)

        query = np.array([1.0, 0.0], dtype=np.float32)
        results = searcher.search(query, k=4)

        scores = [score for _, score in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))