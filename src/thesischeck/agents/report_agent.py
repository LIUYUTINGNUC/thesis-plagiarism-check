"""报告生成 Agent——使用 LLM 生成自然语言查重报告。

LLM 生成的报告比模板拼接更自然、更有洞察力，
能够根据检测数据给出定制化的分析和建议。
"""

from __future__ import annotations

from typing import Any, Optional

from thesischeck.agents.base import BaseAgent, AnalysisResult
from thesischeck.llm import LLMClient

SYSTEM_PROMPT = """你是一个专业的学术查重报告撰写专家。请基于多维度的检测数据
生成全面、客观、有洞察力的查重报告。注意语言客观专业准确。"""


class ReportAgent(BaseAgent):
    """LLM 驱动的报告生成 Agent。"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__(llm_client)

    def generate_report(
        self,
        results: dict[str, Any],
        report_format: str = "markdown",
    ) -> AnalysisResult:
        """基于检测结果生成自然语言报告。

        Args:
            results: 检测结果字典，包含各维度评分和详情。
            report_format: 报告格式（markdown / html / plain）。

        Returns:
            AnalysisResult，包含生成报告文本。
        """
        if not self.llm_available:
            return AnalysisResult(
                success=False,
                error="LLM 不可用，请使用模板报告",
                llm_used=False,
            )

        # 构建检测数据摘要
        summary = self._build_summary(results)

        format_instructions = {
            "markdown": "请用 Markdown 格式输出报告。",
            "html": "请用 HTML 格式输出报告，包含合适的样式。",
            "plain": "请用纯文本格式输出报告。",
        }
        fmt_instruction = format_instructions.get(report_format, "请用 Markdown 格式输出。")

        prompt = f"""请基于以下检测数据生成查重分析报告。

{summary}

{fmt_instruction}

报告要求：
1. 开头：简要概述检测结论
2. 语义相似度分析：解读数值含义，指出高相似段落
3. 观点抄袭分析：评估知识结构相似度
4. AI 生成内容检测：解读 AI 评分
5. 整体评估：综合结论和建议
6. 注意使用专业客观的语言，避免武断结论"""

        try:
            response = self._call_llm(prompt, system=SYSTEM_PROMPT, temperature=0.4)
            return AnalysisResult(
                success=True,
                data={"report": response, "format": report_format},
                llm_used=True,
            )
        except Exception as e:
            return AnalysisResult(
                success=False,
                error=f"报告生成失败: {str(e)}",
                llm_used=True,
            )

    def generate_recommendations(
        self,
        results: dict[str, Any],
    ) -> AnalysisResult:
        """生成针对性的修改建议。"""
        if not self.llm_available:
            return AnalysisResult(success=False, error="LLM 不可用")

        prompt = f"""基于以下查重结果，给出具体的修改建议：

- 语义相似度: {results.get('semantic_similarity', 'N/A')}
- 观点抄袭评分: {results.get('kgraph_score', 'N/A')}
- AI 生成概率: {results.get('ai_score', 'N/A')}
- 学科: {results.get('discipline', 'default')}

请给出 3-5 条具体的、可操作的修改建议。"""

        try:
            response = self._call_llm(prompt, temperature=0.4)
            return AnalysisResult(
                success=True,
                data={"recommendations": response},
                llm_used=True,
            )
        except Exception as e:
            return AnalysisResult(
                success=False, error=str(e), llm_used=True,
            )

    @staticmethod
    def _build_summary(results: dict[str, Any]) -> str:
        """构建检测数据的结构化摘要。"""
        lines = [
            "【检测数据摘要】",
            f"学科配置: {results.get('discipline', 'default')}",
            f"综合相似度: {results.get('overall_score', 0):.2%}",
            f"判定结论: {results.get('overall_verdict', 'unknown')}",
            "",
            "【语义分析】",
            f"语义相似度: {results.get('semantic_similarity', 0):.2%}",
            f"观点抄袭评分: {results.get('kgraph_score', 0):.2%}",
            f"字面相似度: {results.get('literal_similarity', 0):.2%}",
            "",
            "【AI 检测】",
            f"AI 生成概率: {results.get('ai_score', 0):.2%}",
            f"AI 判定: {results.get('ai_verdict', 'not_checked')}",
            "",
            "【判定说明】",
            "综合相似度 > 0.75: 高度相似 / > 0.6: 中度相似 / > 0.45: 轻度相似 / <= 0.45: 不相似",
        ]

        matches = (
            results.get("details", {})
            .get("semantic", {})
            .get("top_matches", [])
        )
        if matches:
            lines.append("")
            lines.append("【高相似段落（前3条）】")
            for i, m in enumerate(matches[:3], 1):
                lines.append(
                    f"  {i}. 相似度 {m.get('similarity', 0):.2%}: "
                    f"{m.get('suspect_sentence', '')[:60]}..."
                )

        return "\n".join(lines)


__all__ = ["ReportAgent"]