"""Semantic similarity computation using cosine similarity and FAISS indexing."""

from __future__ import annotations

from typing import Optional

import numpy as np


# ======================================================================
# Similarity functions
# ======================================================================


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    .. math:: similarity = (v1 * v2) / (||v1|| * ||v2||)

    Args:
        v1: First vector (1-D).
        v2: Second vector (1-D).

    Returns:
        Cosine similarity in ``[-1, 1]``.  Zero vectors return 0.0.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Each row of the input matrix is treated as a vector; the result
    ``[i, j]`` is the cosine similarity between row *i* and row *j*.

    Uses L2-normalised dot product for efficiency.

    Args:
        matrix: ``(n_vectors, dim)`` array.

    Returns:
        ``(n_vectors, n_vectors)`` similarity matrix.
    """
    # L2 normalise each row (avoid division by zero)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized = matrix / norms

    return normalized @ normalized.T


def semantic_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute semantic distance between two vectors.

    .. math:: distance = 1 - cosine\\_similarity(v1, v2)

    Args:
        v1: First vector (1-D).
        v2: Second vector (1-D).

    Returns:
        Distance in ``[0, 2]``; 0 means identical, 2 means opposite.
    """
    sim = cosine_similarity(v1, v2)
    return float(np.clip(1.0 - sim, 0.0, 2.0))


# ======================================================================
# FAISS-based semantic search
# ======================================================================


class SemanticSearcher:
    """Fast similarity search using a FAISS index.

    Wraps ``IndexFlatIP`` (inner product) on L2-normalised vectors, which
    is equivalent to cosine similarity.

    FAISS is imported lazily because it is a heavy native dependency.
    """

    def __init__(self) -> None:
        self._index: Optional["faiss.Index"] = None
        self._dim: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, vectors: np.ndarray) -> None:
        """Build a FAISS index from a collection of vectors.

        Vectors are L2-normalised so that inner-product search yields
        cosine-similarity scores.

        Args:
            vectors: ``(n_vectors, dim)`` array.  An empty array resets the
                index.
        """
        if len(vectors) == 0:
            self._index = None
            self._dim = 0
            return

        import faiss

        self._dim = vectors.shape[1]
        normalized = vectors.astype(np.float32)
        faiss.normalize_L2(normalized)

        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(normalized)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Search for the top-*k* most similar vectors.

        Args:
            query_vector: 1-D or 2-D query array.  If 1-D it is reshaped to
                ``(1, dim)`` internally.
            k: Number of nearest neighbours to return.  Automatically clamped
                to the index size.

        Returns:
            List of ``(index, similarity_score)`` tuples sorted by descending
            score.  Returns an empty list when the index is empty.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        import faiss

        k = min(k, self._index.ntotal)

        query = query_vector.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        faiss.normalize_L2(query)

        distances, indices = self._index.search(query, k)

        results: list[tuple[int, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # -1 is padding from FAISS
                results.append((int(idx), float(dist)))

        return results

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to an existing index.

        If no index exists yet, a new one is built automatically.

        Args:
            vectors: ``(n_vectors, dim)`` array to add.
        """
        if self._index is None:
            self.build_index(vectors)
            return

        import faiss

        normalized = vectors.astype(np.float32)
        faiss.normalize_L2(normalized)
        self._index.add(normalized)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def index_size(self) -> int:
        """Number of vectors currently indexed."""
        if self._index is None:
            return 0
        return self._index.ntotal