"""语义分析 Agent——使用 LLM 进行深度语义比对。

与纯向量方法不同，LLM Agent 能够理解：
- 同义替换（"治疗" ↔ "疗法"）
- 结构重组（主动改被动）
- 观点浓缩与展开
- 跨语言观点抄袭
"""

from __future__ import annotations

import json
import re
from typing import Optional

from thesischeck.agents.base import BaseAgent, AnalysisResult
from thesischeck.llm import LLMClient

SYSTEM_PROMPT = """你是一个专业的论文查重分析专家。你的任务是深度理解两段文本的语义，
判断它们是否表达相同或相似的观点，识别改写式抄袭。

请严格按以下 JSON 格式输出分析结果，不要输出任何其他内容：
{
    "semantic_similarity": 0.0-1.0,
    "opinion_plagiarism": true/false,
    "matched_ideas": [{"original": "...", "suspect": "...", "similarity": 0.0-1.0}],
    "analysis": "分析说明",
    "key_evidence": ["证据1", "证据2"]
}
"""


class SemanticAgent(BaseAgent):
    """LLM 驱动的语义分析 Agent。"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__(llm_client)

    def compare(self, original_text: str, suspect_text: str) -> AnalysisResult:
        """使用 LLM 深度比对两段文本的语义相似度。

        Args:
            original_text: 原始论文文本。
            suspect_text: 待检测文本。

        Returns:
            AnalysisResult，包含语义相似度、匹配观点和分析说明。
        """
        if not self.llm_available:
            return AnalysisResult(
                success=False,
                error="LLM 不可用，请使用统计方法降级",
                llm_used=False,
            )

        prompt = f"""请比对以下两篇论文片段的语义相似度：

【原文】
{original_text[:4000]}

【待检测文】
{suspect_text[:4000]}

请分析：
1. 它们是否表达了相同或相似的观点
2. 是否属于改写式抄袭（用不同词汇表达相同观点）
3. 逐条列出相似的观点对
4. 给出综合语义相似度评分（0-1）
5. 提供关键判断依据"""

        try:
            response = self._call_llm(prompt, system=SYSTEM_PROMPT, temperature=0.2)
            data = self._parse_json_response(response)

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
                error=f"LLM 分析失败: {str(e)}",
                llm_used=True,
            )

    def detect_cross_lingual_plagiarism(
        self, text_a: str, text_b: str,
    ) -> AnalysisResult:
        """检测跨语言抄袭（如英文原文被翻译为中文）。"""
        if not self.llm_available:
            return AnalysisResult(success=False, error="LLM 不可用")

        prompt = f"""请检测以下两段文本是否存在跨语言抄袭关系：

【文本A】
{text_a[:3000]}

【文本B】
{text_b[:3000]}

判断文本B是否是文本A的翻译/意译。给出证据。"""

        try:
            response = self._call_llm(
                prompt,
                system=SYSTEM_PROMPT,
                temperature=0.2,
            )
            data = self._parse_json_response(response)
            return AnalysisResult(
                success=True,
                data=data or {"raw_analysis": response},
                llm_used=True,
            )
        except Exception as e:
            return AnalysisResult(
                success=False, error=str(e), llm_used=True,
            )

    @staticmethod
    def _parse_json_response(response: str) -> Optional[dict]:
        """从 LLM 回复中提取 JSON。"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试提取代码块中的 JSON
        json_match = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```", response
        )
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取最外层花括号
        brace_match = re.search(r"\{[\s\S]*\}", response)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None


__all__ = ["SemanticAgent"]