"""知识图谱 Agent——使用 LLM 提取和比对学术实体与关系。

LLM 能够理解复杂的学术概念和关系，比基于共现的规则方法
更准确地构建知识图谱。
"""

from __future__ import annotations

import json
import re
from typing import Optional

from thesischeck.agents.base import BaseAgent, AnalysisResult
from thesischeck.llm import LLMClient
from thesischeck.core.semantic.kgraph import load_domain_terms

SYSTEM_PROMPT = """你是一个知识图谱构建专家。请从学术文本中提取实体和关系，
然后对比两个文本的知识结构相似度。

请严格按以下 JSON 格式输出：
{
    "original_entities": [{"name": "...", "type": "method/concept/theory/data/..."}],
    "original_relations": [{"source": "...", "target": "...", "relation": "..."}],
    "suspect_entities": [...],
    "suspect_relations": [...],
    "entity_overlap": 0.0-1.0,
    "relation_similarity": 0.0-1.0,
    "structure_similarity": 0.0-1.0,
    "overall_kg_score": 0.0-1.0,
    "analysis": "结构化分析说明"
}
"""


class KGraphAgent(BaseAgent):
    """LLM 驱动的知识图谱 Agent。"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__(llm_client)

    def extract_and_compare(
        self,
        original_text: str,
        suspect_text: str,
        domain: str = "default",
    ) -> AnalysisResult:
        """提取并比对两段文本的知识图谱。

        Args:
            original_text: 原始论文文本。
            suspect_text: 待检测文本。
            domain: 学科领域。

        Returns:
            AnalysisResult，包含实体重叠、关系相似度等。
        """
        if not self.llm_available:
            return AnalysisResult(
                success=False,
                error="LLM 不可用，请使用统计方法降级",
                llm_used=False,
            )

        domain_terms = load_domain_terms(domain)
        terms_str = ", ".join(domain_terms[:30])

        prompt = f"""请分析以下两篇论文的知识结构，提取关键实体和关系。

领域关键词参考：{terms_str}

【原文】
{original_text[:3000]}

【待检测文】
{suspect_text[:3000]}

请分别提取两篇论文的：
1. 核心实体（概念、方法、理论、数据等）
2. 实体间的关系
3. 论证结构

然后比较它们的知识结构相似度。"""

        try:
            response = self._call_llm(prompt, system=SYSTEM_PROMPT, temperature=0.2)
            data = self._parse_json(response)

            if data is None:
                return AnalysisResult(
                    success=False,
                    error="LLM 返回格式无法解析",
                    data={"raw_response": response},
                    llm_used=True,
                )

            return AnalysisResult(
                success=True,
                data=data,
                llm_used=True,
            )

        except Exception as e:
            return AnalysisResult(
                success=False,
                error=f"LLM 知识图谱分析失败: {str(e)}",
                llm_used=True,
            )

    def extract_entities_from_text(
        self, text: str, domain: str = "default",
    ) -> AnalysisResult:
        """仅提取文本中的实体（轻量操作）。"""
        if not self.llm_available:
            return AnalysisResult(success=False, error="LLM 不可用")

        prompt = f"""请从以下学术文本中提取关键实体：

{text[:4000]}

返回 JSON 格式：{{"entities": [{{"name": "...", "type": "..."}}]}}
要求提取 5-15 个最重要的实体。"""

        try:
            response = self._call_llm(prompt, temperature=0.2)
            data = self._parse_json(response)
            return AnalysisResult(
                success=True,
                data=data or {},
                llm_used=True,
            )
        except Exception as e:
            return AnalysisResult(
                success=False, error=str(e), llm_used=True,
            )

    @staticmethod
    def _parse_json(response: str) -> Optional[dict]:
        """从 LLM 回复中提取 JSON。"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        brace_match = re.search(r"\{[\s\S]*\}", response)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        return None


__all__ = ["KGraphAgent"]