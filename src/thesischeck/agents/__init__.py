"""LLM Agent 层——基于大语言模型的论文查重分析 Agent。

每个 Agent 封装一个独立的分析能力，使用 LLM 进行深度语义理解。
当 LLM 不可用时，自动降级到统计方法。
"""

from thesischeck.agents.base import BaseAgent, AnalysisResult
from thesischeck.agents.semantic_agent import SemanticAgent
from thesischeck.agents.kgraph_agent import KGraphAgent
from thesischeck.agents.ai_detection_agent import AIDetectionAgent
from thesischeck.agents.report_agent import ReportAgent

__all__ = [
    "BaseAgent", "AnalysisResult",
    "SemanticAgent", "KGraphAgent",
    "AIDetectionAgent", "ReportAgent",
]