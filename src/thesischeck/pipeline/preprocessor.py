"""文本预处理模块——对论文文本进行规范化处理以备分析。"""

from __future__ import annotations

import re


class TextPreprocessor:
    """论文文本预处理器。

    负责清理、分段、引用移除和参考文献提取等预处理操作。
    """

    # 常见的章节标题模式
    SECTION_PATTERNS: dict[str, list[str]] = {
        "abstract": [
            "abstract", "摘要", "提要", "内容提要",
        ],
        "introduction": [
            "introduction", "introduction and background", "background",
            "引言", "绪论", "引言与背景", "研究背景", "前言",
        ],
        "methods": [
            "methods", "methodology", "method", "materials and methods",
            "research design", "methods and materials",
            "方法", "研究方法", "材料与方法", "研究方法论", "实验方法",
        ],
        "results": [
            "results", "findings", "research results", "experimental results",
            "结果", "研究结果", "实验结果", "结果与分析",
        ],
        "discussion": [
            "discussion", "discussion and analysis", "analysis",
            "讨论", "分析与讨论", "讨论与分析",
        ],
        "conclusion": [
            "conclusion", "conclusions", "summary and conclusion",
            "总结", "结论", "结论与展望", "研究结论",
        ],
    }

    # 引用标记模式（中英文）
    CITATION_PATTERNS = [
        r"\[\d+(?:[,\-\s]+\d+)*\]",          # [1], [1,2], [1, 2], [1-3]
        r"\([\w\s,.;]+\s+(?:et al\.?\s+)?\d{4}[a-z]?\)",  # (Author, 2020)
        r"[一-鿿]+\s+等\.?\s*\(\d{4}[a-z]?\)",    # (作者等, 2020)
        r"[一-鿿]+\s+et al\.?\s*\(\d{4}[a-z]?\)",  # (Author et al., 2020)
        r"\d+\.\s+\w+(?:\s+\w+)+\s+\d{4}",  # 1. Author Name 2020 (reference list)
    ]

    def clean_text(self, text: str) -> str:
        """清理文本：去除多余空白、规范化编码。

        Args:
            text: 原始输入文本。

        Returns:
            清理后的文本。
        """
        if not text:
            return ""

        # 替换不同换行符为统一格式
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 去除页眉页脚常见模式（简单启发式）
        lines = text.split("\n")
        cleaned_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            # 跳过过短的行（可能是页眉页脚或页码）
            if stripped and len(stripped) > 3 and not stripped.isdigit():
                cleaned_lines.append(stripped)

        text = "\n".join(cleaned_lines)
        # 规范化空白字符
        text = re.sub(r"[ \t]+", " ", text)
        # 合并被换行截断的段落
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        # 压缩多个空行为一个
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def segment_sections(self, text: str) -> dict[str, str]:
        """将文本分割为不同章节。

        Args:
            text: 预处理后的论文文本。

        Returns:
            dict，键为章节名称，值为章节内容。
        """
        sections: dict[str, str] = {}
        lines = text.split("\n")
        current_section = "preamble"
        current_content: list[str] = []

        for line in lines:
            stripped = line.strip()
            found_section = None

            # 检查是否为章节标题
            # 先去除编号前缀（如 "1.", "2.1", "I.", "第一章" 等）
            cleaned_line = re.sub(r"^(?:\d+(?:\.\d+)*[\.\s]+|[IVXLCDM]+[\.\s]+|第[一二三四五六七八九十]+[章节部篇]?\s*)", "", stripped.lower().strip()).strip()

            for section_name, patterns in self.SECTION_PATTERNS.items():
                for pattern in patterns:
                    if cleaned_line.startswith(pattern):
                        found_section = section_name
                        break
                if found_section:
                    break

            if found_section:
                # 保存上一个章节
                if current_content:
                    sections[current_section] = "\n".join(
                        current_content
                    ).strip()
                current_section = found_section
                current_content = []
                # 跳过标题行本身（但如果是带序号的标题，保留内容）
                title_cleaned = re.sub(r"^\d+[\.\s]*", "", stripped)
                if title_cleaned.lower() in [
                    p for pats in self.SECTION_PATTERNS.values()
                    for p in pats
                ]:
                    continue
                if stripped:
                    current_content.append(stripped)
            else:
                if stripped:
                    current_content.append(stripped)

        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    @staticmethod
    def remove_citations(text: str) -> str:
        """用占位符替换引用标记。

        Args:
            text: 输入文本。

        Returns:
            引用标记被替换为 [CITATION] 的文本。
        """
        result = text
        for pattern in TextPreprocessor.CITATION_PATTERNS:
            result = re.sub(pattern, "[CITATION]", result)
        return result

    @staticmethod
    def extract_references(text: str) -> list[str]:
        """从文本中提取参考文献列表。

        Args:
            text: 预处理后的论文文本。

        Returns:
            参考文献条目列表。如果未找到则返回空列表。
        """
        references: list[str] = []
        lines = text.split("\n")
        in_references = False

        ref_headers = [
            "references", "bibliography", "works cited",
            "参考文献", "参考书目", "引用文献",
        ]

        for line in lines:
            stripped = line.strip()
            if stripped.lower() in ref_headers:
                in_references = True
                continue

            if in_references:
                if not stripped:
                    # 空行可能表示参考文献结束
                    if references and len(stripped) == 0:
                        break
                    continue
                references.append(stripped)

        return references

    def prepare_for_analysis(self, text: str) -> dict:
        """完整的预处理流程。

        Args:
            text: 原始输入文本。

        Returns:
            dict，包含清理后的文本、章节分段、引用去除后的文本、
            参考文献列表和元数据。
        """
        cleaned = self.clean_text(text)
        sections = self.segment_sections(cleaned)
        no_citations = self.remove_citations(cleaned)
        references = self.extract_references(cleaned)

        # 基本元数据
        words = cleaned.split()
        word_count = len(words)

        return {
            "cleaned_text": cleaned,
            "sections": sections,
            "text_without_citations": no_citations,
            "references": references,
            "metadata": {
                "word_count": word_count,
                "sentence_count": len(re.split(r"[。！？.!?\n]+", cleaned)),
                "section_count": len(sections),
            },
        }


__all__ = ["TextPreprocessor"]
