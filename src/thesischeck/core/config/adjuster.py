"""动态阈值调整模块——基于文本特征自动调整查重参数。"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from thesischeck.core.config.models import DisciplineConfig


class DynamicThresholdAdjuster:
    """基于文本元数据动态调整学科配置阈值。

    不同长度、不同引用密度的文本，其合适的查重阈值也不同。
    此类根据文本的实际特征对基础配置进行微调。
    """

    def __init__(self, base_config: DisciplineConfig):
        self._base = base_config

    def adjust_thresholds(
        self,
        text_length: int = 0,
        citation_density: float = 0.0,
        reference_count: int = 0,
    ) -> DisciplineConfig:
        """根据文本统计信息调整阈值。

        Args:
            text_length: 文本总词数。
            citation_density: 引用占比。
            reference_count: 参考文献数量。

        Returns:
            调整后的 DisciplineConfig 实例。
        """
        cfg = self._base.model_copy(deep=True)

        self._citation_adjustment(cfg, citation_density)
        self._novelty_adjustment(cfg, reference_count)
        self._length_adjustment(cfg, text_length)

        return cfg

    def _citation_adjustment(self, cfg: DisciplineConfig, citation_density: float) -> None:
        """根据引用密度调整引用阈值。

        高引用密度的文本允许更高的引用占比。
        """
        if citation_density <= 0:
            return

        # 基础引用上限 + 密度修正
        base_limit = self._base.citation.max_quotation_ratio
        adjustment = citation_density * 0.15
        cfg.citation.max_quotation_ratio = min(1.0, base_limit + adjustment)

        # 引用密度高时，降低字面相似度阈值（减少误报）
        cfg.similarity.literal_threshold = min(
            1.0,
            self._base.similarity.literal_threshold + citation_density * 0.1,
        )

    def _novelty_adjustment(self, cfg: DisciplineConfig, reference_count: int) -> None:
        """根据参考文献数量调整新发现要求。

        参考文献多的文本（如综述）应降低新发现占比要求。
        """
        if reference_count <= 0:
            return

        # 每10篇参考文献降低0.02的新发现要求，最低不低于0.1
        reduction = min(0.02 * (reference_count // 10), 0.3)
        cfg.citation.min_novel_finding_ratio = max(
            0.1,
            self._base.citation.min_novel_finding_ratio - reduction,
        )

    def _length_adjustment(self, cfg: DisciplineConfig, word_count: int) -> None:
        """根据文本长度调整相似度阈值。

        短文（<1000词）需要更高阈值以避免随机相似导致的误判。
        长文（>10000词）可适当降低阈值以捕捉稀释的抄袭。
        """
        if word_count <= 0:
            return

        if word_count < 1000:
            # 短文：提高阈值
            boost = (1000 - word_count) / 1000 * 0.05
            cfg.similarity.semantic_threshold = min(
                1.0,
                self._base.similarity.semantic_threshold + boost,
            )
        elif word_count > 10000:
            # 长文：降低阈值
            reduction = min((word_count - 10000) / 10000 * 0.05, 0.1)
            cfg.similarity.semantic_threshold = max(
                0.3,
                self._base.similarity.semantic_threshold - reduction,
            )


def calculate_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """计算相似度得分的置信区间。

    Args:
        scores: 相似度得分列表。
        confidence: 置信水平，默认 0.95。

    Returns:
        (下限, 上限) 元组。
    """
    if not scores:
        return (0.0, 0.0)

    arr = np.array(scores)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if len(scores) > 1 else 0.0

    if std == 0:
        return (float(mean), float(mean))

    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_scores.get(confidence, 1.960)

    margin = z * std / math.sqrt(len(scores))
    lower = max(0.0, mean - margin)
    upper = min(1.0, mean + margin)

    return (float(lower), float(upper))


def get_threshold_recommendation(
    config: DisciplineConfig,
    text_length: int,
    references_count: int,
) -> dict:
    """给出基于文本特征的阈值推荐。

    Args:
        config: 基础学科配置。
        text_length: 文本长度（词数）。
        references_count: 参考文献数量。

    Returns:
        dict，包含各项推荐阈值及理由。
    """
    adjuster = DynamicThresholdAdjuster(config)
    adjusted = adjuster.adjust_thresholds(
        text_length=text_length,
        reference_count=references_count,
    )

    return {
        "semantic_threshold": adjusted.similarity.semantic_threshold,
        "literal_threshold": adjusted.similarity.literal_threshold,
        "max_quotation_ratio": adjusted.citation.max_quotation_ratio,
        "min_novel_finding_ratio": adjusted.citation.min_novel_finding_ratio,
        "adjustment_reasoning": {
            "length_factor": (
                "short_text_increased_threshold" if text_length < 1000
                else "long_text_decreased_threshold" if text_length > 10000
                else "standard_threshold"
            ),
            "citation_factor": (
                "increased_quotation_limit" if references_count > 20
                else "standard_quotation_limit"
            ),
        },
    }


__all__ = [
    "DynamicThresholdAdjuster",
    "calculate_confidence_interval",
    "get_threshold_recommendation",
]