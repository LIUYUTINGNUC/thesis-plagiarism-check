"""主流水线编排器——协调各模块完成完整的查重检测流程。

支持双模式：
1. LLM Agent 模式：优先使用大语言模型进行深度语义分析
2. 统计模式：使用 BERT + FAISS + 知识图谱的纯统计方法（降级兜底）
当 LLM 不可用时自动切换到统计模式。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from thesischeck.cache import VectorCache
from thesischeck.core.ai_detection.coherence import coherence_anomaly_score
from thesischeck.core.ai_detection.features import extract_all_features
from thesischeck.core.ai_detection.fingerprint import (
    ai_detection_report,
    combined_ai_score,
)
from thesischeck.core.config.adjuster import DynamicThresholdAdjuster
from thesischeck.core.config.loader import load_discipline_config
from thesischeck.core.config.models import CheckResult, DisciplineConfig
from thesischeck.core.semantic.encoder import SentenceEncoder
from thesischeck.core.semantic.kgraph import DomainKnowledgeGraph
from thesischeck.core.semantic.similarity import (
    SemanticSearcher,
)
from thesischeck.pipeline.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class SemanticResult:
    """语义分析结果。"""
    similarity_score: float
    kgraph_score: float
    literal_score: float
    top_matches: list[dict] = field(default_factory=list)
    llm_used: bool = False


@dataclass
class AIResult:
    """AI检测结果。"""
    ai_score: float
    ai_verdict: str
    features: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    llm_used: bool = False


class PlagiarismChecker:
    """查重检测主编排器。

    支持 LLM Agent 和统计方法双模式：
    - LLM 可用时：使用大模型进行深度语义理解
    - LLM 不可用时：自动降级到 BERT + FAISS + 知识图谱的统计方法
    """

    def __init__(
        self,
        discipline: str = "default",
        use_cache: bool = True,
        encoder_model: str = "all-MiniLM-L6-v2",
        llm_client: Any = None,
    ):
        """初始化查重检测器。

        Args:
            discipline: 学科名称，用于加载对应的配置。
            use_cache: 是否启用向量缓存。
            encoder_model: 语义编码器模型名称。
            llm_client: LLM 客户端实例（为 None 时尝试从环境变量创建）。
        """
        self.discipline_name = discipline
        self.config = load_discipline_config(discipline)
        self.preprocessor = TextPreprocessor()
        self.encoder = SentenceEncoder(
            model_name=encoder_model,
            cache=VectorCache() if use_cache else None,
        )
        self.searcher = SemanticSearcher()

        # 初始化 LLM Agent（尝试加载）
        self._llm = llm_client
        self._agents: dict[str, Any] = {}
        self._init_llm_agents()

    def _init_llm_agents(self) -> None:
        """初始化 LLM Agent（如果 LLM 客户端可用）。"""
        # 如果外部没有传入 LLM client，尝试从环境变量创建
        if self._llm is None and os.getenv("LLM_PROVIDER"):
            try:
                from thesischeck.llm.factory import create_llm_client
                self._llm = create_llm_client()
            except Exception as e:
                logger.warning(f"LLM 客户端初始化失败，将使用统计模式: {e}")

        if self._llm is not None and self._llm.check_available():
            try:
                from thesischeck.agents.ai_detection_agent import AIDetectionAgent
                from thesischeck.agents.kgraph_agent import KGraphAgent
                from thesischeck.agents.report_agent import ReportAgent
                from thesischeck.agents.semantic_agent import SemanticAgent

                self._agents = {
                    "semantic": SemanticAgent(self._llm),
                    "kgraph": KGraphAgent(self._llm),
                    "ai": AIDetectionAgent(self._llm),
                    "report": ReportAgent(self._llm),
                }
                logger.info("LLM Agent 就绪，使用智能检测模式")
            except ImportError as e:
                logger.warning(f"LLM Agent 模块导入失败: {e}")

    @property
    def llm_mode(self) -> bool:
        """是否正在使用 LLM 模式。"""
        return len(self._agents) > 0

    def check(
        self,
        original_text: str,
        suspect_text: str,
        use_llm: Optional[bool] = None,
    ) -> CheckResult:
        """执行完整的查重检测。

        Args:
            original_text: 原始论文文本。
            suspect_text: 待检测论文文本。
            use_llm: 是否使用 LLM 模式。
                为 None 时自动判断（LLM 可用则用，否则降级）。

        Returns:
            CheckResult，包含所有维度的检测结果。
        """
        should_use_llm = self.llm_mode if use_llm is None else use_llm

        # 1. 预处理
        original_processed = self.preprocessor.prepare_for_analysis(original_text)
        suspect_processed = self.preprocessor.prepare_for_analysis(suspect_text)

        # 2. 语义检查（LLM 优先，统计降级）
        semantic_result = self._semantic_check(
            original_processed["cleaned_text"],
            suspect_processed["cleaned_text"],
            use_llm=should_use_llm,
        )

        # 3. AI 内容检测
        ai_result = self._ai_check(
            suspect_processed["cleaned_text"],
            use_llm=should_use_llm,
        )

        # 4. 学科配置调整
        adjuster = DynamicThresholdAdjuster(self.config)
        adjusted_config = adjuster.adjust_thresholds(
            text_length=suspect_processed["metadata"]["word_count"],
        )

        # 5. 综合评分
        weighted = self._apply_config_adjustments(
            semantic_result, ai_result, adjusted_config,
        )

        # 6. 判定
        verdict = self._generate_verdict(weighted, adjusted_config)

        return CheckResult(
            discipline=self.discipline_name,
            semantic_similarity=round(semantic_result.similarity_score, 4),
            kgraph_score=round(semantic_result.kgraph_score, 4),
            literal_similarity=round(semantic_result.literal_score, 4),
            ai_score=round(ai_result.ai_score, 4),
            ai_verdict=ai_result.ai_verdict,
            overall_score=round(weighted, 4),
            overall_verdict=verdict,
            details={
                "semantic": {
                    "top_matches": semantic_result.top_matches[:5],
                    "kgraph_score": semantic_result.kgraph_score,
                    "literal_score": semantic_result.literal_score,
                    "llm_used": semantic_result.llm_used,
                },
                "ai_detection": ai_result.details,
                "ai_llm_used": ai_result.llm_used,
                "config_used": {
                    "name": adjusted_config.name,
                    "semantic_threshold": adjusted_config.similarity.semantic_threshold,
                    "ai_threshold": adjusted_config.ai_detection.ensemble_threshold,
                },
                "mode": "llm_agent" if semantic_result.llm_used else "statistical",
            },
        )

    def _semantic_check(
        self,
        original_text: str,
        suspect_text: str,
        use_llm: bool = False,
    ) -> SemanticResult:
        """执行语义层面的查重分析。

        优先使用 LLM Agent（如可用），否则使用统计方法。
        """
        # ---- LLM Agent 模式 ----
        if use_llm and "semantic" in self._agents and "kgraph" in self._agents:
            try:
                semantic_agent = self._agents["semantic"]
                kgraph_agent = self._agents["kgraph"]

                # LLM 语义分析
                llm_result = semantic_agent.compare(original_text, suspect_text)
                if llm_result.success:
                    sim_score = llm_result.data.get("semantic_similarity", 0.5)
                    if isinstance(sim_score, (int, float)) and 0 <= sim_score <= 1:
                        # LLM 知识图谱分析
                        kg_result = kgraph_agent.extract_and_compare(
                            original_text, suspect_text, self.discipline_name,
                        )
                        kg_score = 0.0
                        if kg_result.success:
                            kg_score = kg_result.data.get("overall_kg_score", 0.0)

                        # 字面相似度仍用统计方法
                        literal_score = self._literal_similarity(
                            original_text, suspect_text,
                        )

                        logger.info("LLM Agent 语义分析完成")
                        return SemanticResult(
                            similarity_score=float(sim_score),
                            kgraph_score=float(kg_score),
                            literal_score=float(literal_score),
                            llm_used=True,
                        )
            except Exception as e:
                logger.warning(f"LLM 语义分析失败，降级到统计模式: {e}")

        # ---- 统计模式（降级兜底） ----
        original_sentences = self.encoder._split_sentences(original_text)
        suspect_sentences = self.encoder._split_sentences(suspect_text)

        if not original_sentences or not suspect_sentences:
            return SemanticResult(0.0, 0.0, 0.0)

        try:
            orig_vectors = self.encoder.encode(original_sentences)
            susp_vectors = self.encoder.encode(suspect_sentences)
        except Exception:
            return SemanticResult(0.0, 0.0, 0.0)

        self.searcher.build_index(orig_vectors)
        top_matches: list[dict] = []

        for i, sv in enumerate(susp_vectors):
            results = self.searcher.search(sv, k=1)
            if results:
                idx, score = results[0]
                top_matches.append({
                    "suspect_sentence": suspect_sentences[i][:100],
                    "original_sentence": original_sentences[idx][:100],
                    "similarity": round(float(score), 4),
                })

        semantic_score = (
            float(np.mean([m["similarity"] for m in top_matches]))
            if top_matches else 0.0
        )

        kg = DomainKnowledgeGraph()
        kg_score_data = kg.idea_plagiarism_score(
            original_text, suspect_text, self.discipline_name,
        )
        kg_score = float(kg_score_data.get("overall_idea_plagiarism_score", 0.0))
        literal_score = float(self._literal_similarity(original_text, suspect_text))

        return SemanticResult(
            similarity_score=semantic_score,
            kgraph_score=kg_score,
            literal_score=literal_score,
            top_matches=top_matches,
        )

    @staticmethod
    def _literal_similarity(text1: str, text2: str) -> float:
        """计算字面相似度（基于 n-gram 重叠率）。"""
        def _char_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
            chars = text.replace(" ", "")
            return set(
                tuple(chars[i:i + n])
                for i in range(len(chars) - n + 1)
            )

        ngrams1 = _char_ngrams(text1, 3)
        ngrams2 = _char_ngrams(text2, 3)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = ngrams1 & ngrams2
        return float(len(intersection) / len(ngrams1 | ngrams2))

    def _ai_check(self, text: str, use_llm: bool = False) -> AIResult:
        """执行 AI 生成内容检测。

        优先使用 LLM Agent（如可用），否则使用统计方法。
        """
        # ---- LLM Agent 模式 ----
        if use_llm and "ai" in self._agents:
            try:
                ai_agent = self._agents["ai"]
                ai_result = ai_agent.analyze(text)

                if ai_result.success:
                    data = ai_result.data
                    ai_score = float(data.get("ai_score", 0.5))
                    verdict = data.get("verdict", "possibly_ai_generated")

                    logger.info("LLM Agent AI检测完成")
                    return AIResult(
                        ai_score=ai_score,
                        ai_verdict=verdict,
                        details={
                            "evidence": data.get("evidence", {}),
                            "confidence": data.get("confidence", "low"),
                            "explanation": data.get("explanation", ""),
                            "llm_analysis": True,
                        },
                        llm_used=True,
                    )
            except Exception as e:
                logger.warning(f"LLM AI检测失败，降级到统计模式: {e}")

        # ---- 统计模式（降级兜底） ----
        features = extract_all_features(text)
        ai_score = float(combined_ai_score(text))
        ai_report = ai_detection_report(text)

        try:
            coh_anomaly = float(coherence_anomaly_score(text, self.encoder))
        except Exception:
            coh_anomaly = 0.0

        return AIResult(
            ai_score=ai_score,
            ai_verdict=ai_report["verdict"],
            features=features,
            details={
                **ai_report["details"],
                "coherence_anomaly": round(coh_anomaly, 4),
                "llm_analysis": False,
            },
        )

    @staticmethod
    def _apply_config_adjustments(
        semantic: SemanticResult,
        ai: AIResult,
        config: DisciplineConfig,
    ) -> float:
        """根据学科配置对各项得分进行加权综合。"""
        semantic_combined = (
            (1 - config.similarity.kgraph_weight) * semantic.similarity_score
            + config.similarity.kgraph_weight * semantic.kgraph_score
        )
        overall = 0.7 * semantic_combined + 0.3 * ai.ai_score
        return float(min(1.0, overall))

    @staticmethod
    def _generate_verdict(
        weighted_score: float,
        config: DisciplineConfig,
    ) -> str:
        """根据综合得分和学科配置生成判定结论。"""
        threshold = config.similarity.semantic_threshold

        if weighted_score >= threshold + 0.15:
            return "highly_similar"
        elif weighted_score >= threshold:
            return "moderately_similar"
        elif weighted_score >= threshold - 0.15:
            return "slightly_similar"
        else:
            return "distinct"


__all__ = ["PlagiarismChecker", "SemanticResult", "AIResult"]
