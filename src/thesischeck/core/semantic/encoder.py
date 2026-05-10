"""BERT-based sentence encoder for converting text to semantic vectors.

Uses SentenceTransformer models for efficient sentence-level embeddings.
The model is loaded lazily on first encode call to avoid unnecessary memory usage.
"""

import hashlib
import re
from typing import Optional

import numpy as np

from thesischeck.cache import VectorCache  # noqa: F401 - re-exported from __init__


class SentenceEncoder:
    """BERT-based sentence encoder for converting text to semantic vectors.

    Uses SentenceTransformer models for efficient sentence-level embeddings.
    The model is loaded lazily on first ``encode`` call to reduce startup cost.

    Args:
        model_name: Pre-trained SentenceTransformer model name.
            Default ``'all-MiniLM-L6-v2'`` produces 384-dimensional vectors.
        cache: Optional VectorCache for caching embedding results (e.g., Redis).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache: Optional[VectorCache] = None):
        self._model_name = model_name
        self._cache = cache
        self._model = None
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazily load the SentenceTransformer model on first use."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(self._model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using regex on Chinese and English punctuation.

        Handles ``。！？.!?`` and newlines as sentence boundaries.
        Filters out empty / whitespace-only sentences.

        Args:
            text: Raw input text.

        Returns:
            A list of cleaned sentence strings.
        """
        parts = re.split(r"[。！？.!?\n]+", text)
        return [s.strip() for s in parts if s.strip()]

    def _make_cache_key(self, text: str) -> str:
        """Build a deterministic cache key from text content and current model name."""
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"emb:{self._model_name}:{text_hash}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into semantic vectors.

        Args:
            texts: List of strings to encode.

        Returns:
            Numpy array of shape ``(len(texts), embedding_dim)``.
        """
        # --- Empty input ----------------------------------------------------
        if not texts:
            self._load_model()
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # --- Cache-enabled path ---------------------------------------------
        if self._cache is not None:
            results: list[tuple[int, np.ndarray]] = []
            uncached_indices: list[int] = []
            uncached_texts: list[str] = []

            for i, text in enumerate(texts):
                key = self._make_cache_key(text)
                cached = self._cache.get(key)
                if cached is not None:
                    results.append((i, cached))
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

            # Encode only the texts that missed the cache
            if uncached_texts:
                self._load_model()
                encoded = self._model.encode(uncached_texts, show_progress_bar=False)  # type: ignore[union-attr]
                for idx, text, vec in zip(uncached_indices, uncached_texts, encoded):
                    key = self._make_cache_key(text)
                    self._cache.set(key, vec)
                    results.append((idx, vec))

            # Restore original ordering and stack into a 2-D array
            results.sort(key=lambda x: x[0])
            return np.stack([r[1] for r in results])

        # --- No cache — direct encoding -------------------------------------
        self._load_model()
        return self._model.encode(texts, show_progress_bar=False)  # type: ignore[union-attr]

    def encode_sentences(self, text: str) -> dict[str, np.ndarray]:
        """Split text into sentences and encode each one individually.

        Args:
            text: Raw input text.

        Returns:
            Dict mapping each sentence string to its embedding vector.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return {}
        vectors = self.encode(sentences)
        return {sent: vectors[i] for i, sent in enumerate(sentences)}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the loaded model."""
        self._load_model()
        # _dim is guaranteed to be set after _load_model
        assert self._dim is not None
        return self._dim

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name
