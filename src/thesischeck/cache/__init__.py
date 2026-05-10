"""Redis-backed caching layer for embeddings and intermediate results."""

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorCacheProtocol(Protocol):
    """Protocol for vector caching (e.g., Redis-based).

    Defines the interface for caching and retrieving embedding vectors.
    """

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve a cached vector by key."""
        ...

    def set(self, key: str, vector: np.ndarray) -> None:
        """Cache a vector under the given key."""
        ...


# Re-export the concrete implementation
from thesischeck.cache.redis_cache import CacheKeyBuilder, VectorCache  # noqa: E402, F401

__all__ = ["VectorCache", "VectorCacheProtocol", "CacheKeyBuilder"]
__version__ = "0.1.0"
