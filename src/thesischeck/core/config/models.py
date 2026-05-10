"""学科配置的数据模型——使用 Pydantic 定义配置结构和验证规则。"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CitationConfig(BaseModel):
    """引用相关配置。"""

    max_quotation_ratio: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="最大允许引用占比",
    )
    min_novel_finding_ratio: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="最低新发现占比要求",
    )
    self_plagiarism_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="自我抄袭判定阈值",
    )

    @field_validator("min_novel_finding_ratio")
    @classmethod
    def novel_ratio_plus_quotation_ratio_should_not_exceed_1(cls, v, info):
        """新发现占比 + 引用占比不能超过 1（宽松检查）。"""
        quotation_ratio = info.data.get("max_quotation_ratio", 0)
        if v + quotation_ratio > 1.3:
            raise ValueError(
                f"min_novel_finding_ratio ({v}) + max_quotation_ratio "
                f"({quotation_ratio}) 之和不应超过 1.3"
            )
        return v


class SimilarityConfig(BaseModel):
    """相似度检测相关配置。"""

    semantic_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="语义相似度判定阈值（高于此值判定为相似）",
    )
    literal_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="字面相似度判定阈值",
    )
    kgraph_weight: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="知识图谱相似度在综合评分中的权重",
    )


class AIConfig(BaseModel):
    """AI生成内容检测相关配置。"""

    feature_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "entropy": 0.2,
            "coherence": 0.25,
            "burstiness": 0.25,
            "repetition": 0.15,
            "vocabulary": 0.15,
        },
        description="各检测特征的权重",
    )
    ensemble_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="AI判定集成阈值（高于此值判定为AI生成）",
    )


class DisciplineConfig(BaseModel):
    """学科配置的完整模型。"""

    name: str = Field(
        ..., min_length=1,
        description="学科名称标识",
    )
    display_name: str = Field(
        default="",
        description="学科显示名称",
    )
    description: str = Field(
        default="",
        description="学科描述",
    )
    citation: CitationConfig = Field(
        default_factory=CitationConfig,
        description="引用配置",
    )
    similarity: SimilarityConfig = Field(
        default_factory=SimilarityConfig,
        description="相似度检测配置",
    )
    ai_detection: AIConfig = Field(
        default_factory=AIConfig,
        description="AI检测配置",
    )


class CheckResult(BaseModel):
    """一次查重检测的完整结果。"""

    discipline: str = Field(..., description="使用的学科配置")
    semantic_similarity: float = Field(..., ge=0.0, le=1.0)
    kgraph_score: float = Field(default=0.0, ge=0.0, le=1.0)
    literal_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    ai_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ai_verdict: str = Field(default="not_checked")
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_verdict: str = Field(default="unknown")
    details: dict = Field(default_factory=dict)


__all__ = [
    "CitationConfig",
    "SimilarityConfig",
    "AIConfig",
    "DisciplineConfig",
    "CheckResult",
]
