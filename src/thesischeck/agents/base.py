"""Agent 基类和通用数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from thesischeck.llm import LLMClient, Message, Role


@dataclass
class AnalysisResult:
    """Agent 分析结果的通用模型。"""
    success: bool = True
    error: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    llm_used: bool = False
    fallback_used: bool = False


class BaseAgent:
    """Agent 基类。

    所有 Agent 共享以下能力：
    - LLM 驱动的深度分析
    - 统计方法降级兜底
    - 统一的结果模型
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client

    @property
    def llm_available(self) -> bool:
        """检查 LLM 客户端是否可用。"""
        return self._llm is not None and self._llm.check_available()

    def _call_llm(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """调用 LLM 的统一入口。

        Args:
            prompt: 用户提示词。
            system: 系统提示词。
            temperature: 温度参数。

        Returns:
            LLM 回复文本。

        Raises:
            RuntimeError: LLM 不可用时抛出。
        """
        if not self.llm_available:
            raise RuntimeError("LLM 客户端不可用，请检查 API Key 配置")
        return self._llm.chat(
            messages=[Message(role=Role.USER, content=prompt)],
            system=system,
            temperature=temperature,
            **kwargs,
        )

    def _call_llm_structured(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """调用 LLM 的结构化输出接口。"""
        if not self.llm_available:
            raise RuntimeError("LLM 客户端不可用")
        return self._llm.chat_structured(
            messages=[Message(role=Role.USER, content=prompt)],
            system=system,
            **kwargs,
        )


# ======================================================================
# 通用系统提示词模板
# ======================================================================

SYSTEM_PROMPTS: dict[str, str] = {
    "semantic_analysis": """你是一个专业的论文查重分析专家。你的任务是：
1. 深度理解两段文本的语义，判断它们是否表达相同或相似的观点
2. 识别"改写式抄袭"——用不同词汇表达相同观点
3. 区分"合理引用"与"过度依赖"
4. 评估观点的新颖程度

请始终用 JSON 格式输出分析结果。""",

    "kgraph_extraction": """你是一个知识图谱构建专家。你的任务是：
1. 从学术文本中提取关键实体（概念、方法、理论、数据等）
2. 识别实体之间的语义关系
3. 构建实体关系的结构化描述
4. 对比两组实体关系的相似度

请始终用 JSON 格式输出。""",

    "ai_detection": """你是一个 AI 生成内容检测专家。你的任务是：
1. 分析文本的语言特征
2. 判断文本是否由 AI 生成
3. 给出判断依据和置信度
4. 结合 token 概率数据进行分析

请始终用 JSON 格式输出检测结果。""",

    "report_generation": """你是一个学术查重报告撰写专家。你的任务是：
1. 基于多维度检测数据生成全面的查重报告
2. 用自然语言清晰解释各项评分
3. 提供具体的修改建议
4. 注意语言客观、专业、准确""",
}
