"""报告生成模块——将检测结果生成为多种格式的报告。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from thesischeck.core.config.models import CheckResult


class ReportGenerator:
    """查重报告生成器。

    支持文本、HTML 和 JSON 三种输出格式。
    HTML 报告包含内联的 SVG 可视化图表。
    """

    @staticmethod
    def generate_text_report(result: CheckResult) -> str:
        """生成纯文本格式报告。

        Args:
            result: 查重检测结果。

        Returns:
            格式化的文本报告。
        """
        lines = [
            "=" * 60,
            "语义级论文查重检测报告",
            "=" * 60,
            f"检测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"学科配置：{result.discipline}",
            "",
            "【综合评分】",
            f"综合相似度：{result.overall_score:.2%}",
            f"判定结论：{_verdict_label(result.overall_verdict)}",
            "",
            "【语义分析结果】",
            f"语义相似度：{result.semantic_similarity:.2%}",
            f"观点抄袭评分：{result.kgraph_score:.2%}",
            f"字面相似度：{result.literal_similarity:.2%}",
            "",
            "【AI生成内容检测】",
            f"AI生成概率：{result.ai_score:.2%}",
            f"判定：{_ai_label(result.ai_verdict)}",
            "",
        ]

        # 添加 Top 匹配详情
        matches = result.details.get("semantic", {}).get("top_matches", [])
        if matches:
            lines.append("【高相似段落】")
            for i, m in enumerate(matches[:3], 1):
                lines.extend([
                    f"  {i}. 相似度：{m['similarity']:.2%}",
                    f"     原文：{m.get('original_sentence', 'N/A')[:80]}",
                    f"     疑文：{m.get('suspect_sentence', 'N/A')[:80]}",
                ])
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_html_report(result: CheckResult) -> str:
        """生成 HTML 格式报告，带内联可视化。

        Args:
            result: 查重检测结果。

        Returns:
            HTML 字符串。
        """
        score_bars = _score_bars_html(result)
        ai_details = result.details.get("ai_detection", {})
        config_info = result.details.get("config_used", {})

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>论文查重检测报告</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 10px; }}
h2 {{ color: #0f3460; margin-top: 30px; }}
.bar-container {{ background: #f0f0f0; border-radius: 8px; height: 24px; margin: 8px 0; overflow: hidden; }}
.bar {{ height: 100%; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: bold; }}
.high {{ background: linear-gradient(90deg, #e74c3c, #c0392b); }}
.medium {{ background: linear-gradient(90deg, #f39c12, #e67e22); }}
.low {{ background: linear-gradient(90deg, #27ae60, #2ecc71); }}
.ai {{ background: linear-gradient(90deg, #8e44ad, #9b59b6); }}
.metric {{ margin: 15px 0; }}
.metric-label {{ display: flex; justify-content: space-between; font-size: 14px; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
th {{ background: #16213e; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.verdict {{ font-size: 18px; font-weight: bold; padding: 10px 15px; border-radius: 6px; display: inline-block; }}
.verdict-high {{ background: #fde8e8; color: #c0392b; }}
.verdict-medium {{ background: #fef3cd; color: #856404; }}
.verdict-low {{ background: #d4edda; color: #155724; }}
</style>
</head>
<body>
<h1>语义级论文查重检测报告</h1>
<p>检测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p>学科配置：<strong>{result.discipline}</strong></p>

<h2>综合判定</h2>
<div class="verdict {_verdict_class(result.overall_verdict)}">{_verdict_label(result.overall_verdict)}</div>
<p>综合相似度评分：<strong>{result.overall_score:.2%}</strong></p>

<h2>各维度评分</h2>
{score_bars}

<h2>AI生成内容检测</h2>
<table>
<tr><th>指标</th><th>值</th></tr>
<tr><td>AI 生成概率</td><td>{result.ai_score:.2%}</td></tr>
<tr><td>判定结果</td><td>{_ai_label(result.ai_verdict)}</td></tr>
<tr><td>突发性评分</td><td>{ai_details.get('burstiness', 'N/A')}</td></tr>
<tr><td>一元重复率</td><td>{ai_details.get('unigram_repetition', 'N/A')}</td></tr>
<tr><td>二元重复率</td><td>{ai_details.get('bigram_repetition', 'N/A')}</td></tr>
</table>

<h2>高相似段落</h2>
{_matches_html(result)}

<h2>配置信息</h2>
<table>
<tr><th>参数</th><th>值</th></tr>
<tr><td>学科</td><td>{config_info.get('name', 'N/A')}</td></tr>
<tr><td>语义相似度阈值</td><td>{config_info.get('semantic_threshold', 'N/A')}</td></tr>
<tr><td>AI检测阈值</td><td>{config_info.get('ai_threshold', 'N/A')}</td></tr>
</table>
</body>
</html>"""

    @staticmethod
    def generate_json_report(result: CheckResult) -> str:
        """生成 JSON 格式报告。

        Args:
            result: 查重检测结果。

        Returns:
            JSON 字符串。
        """
        report = {
            "report_meta": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0",
            },
            "overall": {
                "score": result.overall_score,
                "verdict": result.overall_verdict,
                "verdict_label": _verdict_label(result.overall_verdict),
            },
            "semantic_analysis": {
                "semantic_similarity": result.semantic_similarity,
                "kgraph_score": result.kgraph_score,
                "literal_similarity": result.literal_similarity,
                "top_matches": result.details.get("semantic", {}).get(
                    "top_matches", []
                ),
            },
            "ai_detection": result.details.get("ai_detection", {}),
            "config": result.details.get("config_used", {}),
            "discipline": result.discipline,
        }
        return json.dumps(report, ensure_ascii=False, indent=2)


# ======================================================================
# 辅助函数
# ======================================================================


def _verdict_label(verdict: str) -> str:
    labels = {
        "highly_similar": "高度相似 — 建议详细审查",
        "moderately_similar": "中度相似 — 需要进一步检查",
        "slightly_similar": "轻度相似 — 可参考",
        "distinct": "不相似 — 通过检测",
        "unknown": "无法判定",
    }
    return labels.get(verdict, verdict)


def _ai_label(verdict: str) -> str:
    labels = {
        "likely_ai_generated": "极可能为AI生成",
        "possibly_ai_generated": "可能为AI生成",
        "possibly_human_written": "可能为人类写作",
        "likely_human_written": "极可能为人类写作",
        "not_checked": "未检测",
    }
    return labels.get(verdict, verdict)


def _verdict_class(verdict: str) -> str:
    mapping = {
        "highly_similar": "verdict-high",
        "moderately_similar": "verdict-medium",
        "slightly_similar": "verdict-low",
        "distinct": "verdict-low",
    }
    return mapping.get(verdict, "verdict-medium")


def _score_bars_html(result: CheckResult) -> str:
    bars: list[str] = []
    scores: list[tuple[str, float, str]] = [
        ("语义相似度", result.semantic_similarity, "high" if result.semantic_similarity > 0.7 else "medium"),
        ("观点抄袭评分", result.kgraph_score, "high" if result.kgraph_score > 0.6 else "medium"),
        ("字面相似度", result.literal_similarity, "high" if result.literal_similarity > 0.8 else "medium"),
        ("AI生成概率", result.ai_score, "ai"),
    ]

    for label, score, bar_class in scores:
        pct = f"{score:.1%}"
        bars.append(f"""<div class="metric">
<div class="metric-label"><span>{label}</span><span>{pct}</span></div>
<div class="bar-container"><div class="bar {bar_class}" style="width:{pct}">{pct}</div></div>
</div>""")

    return "\n".join(bars)


def _matches_html(result: CheckResult) -> str:
    matches = result.details.get("semantic", {}).get("top_matches", [])
    if not matches:
        return "<p>未检测到高相似段落。</p>"

    rows: list[str] = []
    for i, m in enumerate(matches[:5], 1):
        rows.append(f"""<tr>
<td>{i}</td>
<td>{m.get('original_sentence', 'N/A')[:80]}</td>
<td>{m.get('suspect_sentence', 'N/A')[:80]}</td>
<td>{m.get('similarity', 0):.1%}</td>
</tr>""")

    return f"""<table>
<tr><th>#</th><th>原文</th><th>疑文</th><th>相似度</th></tr>
{''.join(rows)}
</table>"""


__all__ = ["ReportGenerator"]