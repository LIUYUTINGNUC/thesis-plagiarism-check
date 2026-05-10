"""基于 Redis 的向量缓存模块。

缓存语义向量以加速重复查询，减少模型推理调用。
支持设置 TTL、批量操作和缓存统计。
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np


class VectorCache:
    """向量缓存，使用 Redis 存储 numpy 数组。

    提供 get/set 接口，支持自动序列化/反序列化 numpy 数组，
    TTL 过期和缓存命中率统计。

    当 Redis 不可用时自动降级为内存缓存（基于 dict）。
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: int = 3600,
        max_memory: str = "512mb",
        use_redis: bool = False,
    ):
        """初始化缓存。

        Args:
            redis_url: Redis 连接 URL。
            ttl: 默认 TTL（秒）。
            max_memory: Redis maxmemory 配置。
            use_redis: 是否使用 Redis，为 False 时使用内存缓存兜底。
        """
        self._ttl = ttl
        self._max_memory = max_memory
        self._redis_url = redis_url
        self._redis: Any = None
        self._memory_cache: dict[str, np.ndarray] = {}
        self._memory_cache_timestamps: dict[str, float] = {}

        self._stats = {"hits": 0, "misses": 0, "sets": 0}

        if use_redis:
            self._connect_redis()

    def _connect_redis(self) -> None:
        """尝试连接 Redis。"""
        try:
            import redis as _redis

            self._redis = _redis.from_url(
                self._redis_url,
                decode_responses=True,
            )
            self._redis.ping()
        except Exception:
            self._redis = None

    @staticmethod
    def _serialize(vector: np.ndarray) -> str:
        """将 numpy 数组序列化为 JSON。"""
        return json.dumps({
            "shape": list(vector.shape),
            "dtype": str(vector.dtype),
            "data": vector.flatten().tolist(),
        })

    @staticmethod
    def _deserialize(data: str) -> np.ndarray:
        """从 JSON 反序列化为 numpy 数组。"""
        obj = json.loads(data)
        arr = np.array(obj["data"], dtype=np.dtype(obj["dtype"]))
        return arr.reshape(obj["shape"])

    def get(self, key: str) -> Optional[np.ndarray]:
        """从缓存获取向量。

        Args:
            key: 缓存键。

        Returns:
            numpy 数组，如果未命中返回 None。
        """
        if self._redis is not None:
            try:
                data = self._redis.get(key)
                if data is not None:
                    self._stats["hits"] += 1
                    return self._deserialize(data)
            except Exception:
                pass

        # 内存缓存兜底
        if key in self._memory_cache:
            self._stats["hits"] += 1
            return self._memory_cache[key]

        self._stats["misses"] += 1
        return None

    def set(self, key: str, vector: np.ndarray, ttl: Optional[int] = None) -> None:
        """将向量存入缓存。

        Args:
            key: 缓存键。
            vector: 要缓存的 numpy 数组。
            ttl: 过期时间（秒），为 None 时使用默认值。
        """
        ttl = ttl or self._ttl
        serialized = self._serialize(vector)

        if self._redis is not None:
            try:
                self._redis.setex(key, ttl, serialized)
                self._stats["sets"] += 1
                return
            except Exception:
                pass

        # 内存缓存兜底
        self._memory_cache[key] = vector
        self._stats["sets"] += 1

    def get_batch(self, keys: list[str]) -> list[Optional[np.ndarray]]:
        """批量获取缓存向量。

        Args:
            keys: 缓存键列表。

        Returns:
            numpy 数组列表，未命中的位置为 None。
        """
        return [self.get(key) for key in keys]

    def set_batch(
        self, items: list[tuple[str, np.ndarray]],
        ttl: Optional[int] = None,
    ) -> None:
        """批量存储向量。

        Args:
            items: (key, vector) 元组列表。
            ttl: 过期时间（秒）。
        """
        for key, vector in items:
            self.set(key, vector, ttl)

    def clear(self) -> None:
        """清空所有缓存。"""
        if self._redis is not None:
            try:
                self._redis.flushdb()
            except Exception:
                pass
        self._memory_cache.clear()
        self._memory_cache_timestamps.clear()

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计信息。

        Returns:
            dict，包含命中数、未命中数、命中率和设置次数。
        """
        total = self._stats["hits"] + self._stats["misses"]
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_ratio": round(
                self._stats["hits"] / total, 4
            ) if total > 0 else 0.0,
            "sets": self._stats["sets"],
            "memory_cache_size": len(self._memory_cache),
            "using_redis": self._redis is not None,
        }


class CacheKeyBuilder:
    """构建标准化的缓存键。"""

    @staticmethod
    def semantic_key(text_hash: str, model_name: str) -> str:
        """构建语义向量缓存键。

        Args:
            text_hash: 文本的 MD5 哈希。
            model_name: 模型名称。

        Returns:
            缓存键字符串。
        """
        return f"emb:{model_name}:{text_hash}"

    @staticmethod
    def graph_key(entity_set: frozenset) -> str:
        """构建知识图谱缓存键。

        Args:
            entity_set: 实体集合。

        Returns:
            缓存键字符串。
        """
        entity_hash = hash(entity_set)
        return f"kg:{entity_hash}"


__all__ = ["VectorCache", "CacheKeyBuilder"]