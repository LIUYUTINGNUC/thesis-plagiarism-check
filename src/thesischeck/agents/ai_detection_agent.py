"""AI 生成内容检测 Agent——使用 LLM 进行深度分析。

结合 LLM 的 token 概率分析、感知能力和统计特征，
提供更准确的 AI 生成内容判定。
"""

from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np

from thesischeck.agents.base import AnalysisResult, BaseAgent
from thesischeck.llm import LLMClient

SYSTEM_PROMPT = """你是一个 AI 生成内容检测专家。请判断一段文本是由人类撰写
还是 AI 生成，并给出详细的分析依据。

请严格按以下 JSON 格式输出（不要输出其他内容）：
{
    "ai_score": 0.0-1.0,
    "verdict": "likely_human_written/possibly_human_written/possibly_ai_generated/likely_ai_generated",
    "confidence": "low/medium/high",
    "evidence": {
        "style_analysis": "...",
        "logic_analysis": "...",
        "knowledge_analysis": "...",
        "suspicious_patterns": ["..."]
    },
    "explanation": "综合判断理由"
}
"""


class AIDetectionAgent(BaseAgent):
    """LLM 驱动的 AI 生成内容检测 Agent。"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__(llm_client)

    def analyze(self, text: str, token_probs: Optional[list[dict]] = None) -> AnalysisResult:
        """使用 LLM 深度分析文本是否为 AI 生成。

        Args:
            text: 待检测文本。
            token_probs: 可选的 token 级概率数据（来自 chat_with_probs）。

        Returns:
            AnalysisResult，包含 AI 评分、判定和证据。
        """
        if not self.llm_available:
            return AnalysisResult(
                success=False,
                error="LLM 不可用，请使用统计方法降级",
                llm_used=False,
            )

        prompt = f"""请分析以下文本是否为 AI 生成内容：

【待检测文本】
{text[:4000]}

请从以下维度分析：
1. 语言风格：是否过于规整、缺少人类写作的多样性
2. 逻辑结构：论证是否自然
3. 知识呈现：是否存在"模糊泛泛而谈"的 AI 典型特征
4. 创意与洞见：是否有真正的原创思考"""

        prob_context = ""
        if token_probs:
            logprobs = [p.get("logprob", 0) for p in token_probs if p.get("logprob") is not None]
            if logprobs:
                mean_lp = float(np.mean(logprobs))
                std_lp = float(np.std(logprobs))
                prob_context = (
                    f"\n额外 Token 概率数据（均值: {mean_lp:.4f}, 标准差: {std_lp:.4f}）：\n"
                    '低概率分布通常表示人类写作的"意外"用词选择。'
                )
        prompt += prob_context

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

            # 融合 token 概率数据（如果有）
            if token_probs and data:
                data["token_probability_data"] = {
                    "num_tokens": len(token_probs),
                }

            return AnalysisResult(
                success=True,
                data=data,
                llm_used=True,
            )

        except Exception as e:
            return AnalysisResult(
                success=False,
                error=f"LLM AI检测失败: {str(e)}",
                llm_used=True,
            )

    def analyze_with_probs(
        self, text: str,
    ) -> AnalysisResult:
        """使用 LLM 的 token 概率 API 进行 AI 检测。

        优先使用 chat_with_probs 获取 token 级概率数据，
        再结合 LLM 的语义理解进行综合判断。
        """
        if not self.llm_available:
            return AnalysisResult(success=False, error="LLM 不可用")

        try:
            # Step 1: 获取 token 概率
            truncated = text[:2000]
            _, probs = self._llm.chat_with_probs(
                messages=[{"role": "user", "content": f"Continue: {truncated}"}]
            )

            # Step 2: LLM 综合分析（含概率数据）
            return self.analyze(text, token_probs=probs)

        except Exception as e:
            return AnalysisResult(
                success=False,
                error=f"Token 概率分析失败: {str(e)}",
                llm_used=True,
            )

    @staticmethod
    def _parse_json(response: str) -> Optional[dict]:
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


__all__ = ["AIDetectionAgent"]
