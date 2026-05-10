"""逻辑连贯性检测模块——用于AI生成内容检测。

分析文本的逻辑连贯性特征，包括过渡词使用、局部/全局连贯性、
以及篇章结构模式。AI生成文本在这些维度上往往与人类写作存在差异。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from thesischeck.core.semantic.encoder import SentenceEncoder


# ======================================================================
# 过渡词分类
# ======================================================================

TRANSITION_WORDS: dict[str, list[str]] = {
    "additive": [
        "and", "also", "furthermore", "moreover", "besides", "in addition",
        "additionally", "likewise", "similarly", "as well as", "not only",
        "further", "plus", "additionally",
        # 中文
        "而且", "此外", "另外", "还有", "再者", "加之", "同时",
        "并且", "以及", "不仅", "同样",
    ],
    "adversative": [
        "but", "however", "nevertheless", "nonetheless", "yet", "although",
        "though", "even though", "despite", "in spite of", "on the other hand",
        "conversely", "whereas", "while", "instead", "rather",
        # 中文
        "但是", "然而", "可是", "不过", "虽然", "尽管", "却",
        "反之", "另一方面", "相反", "而",
    ],
    "causal": [
        "because", "since", "as", "therefore", "thus", "hence", "consequently",
        "accordingly", "as a result", "due to", "owing to", "for this reason",
        "lead to", "result in", "cause", "so",
        # 中文
        "因为", "所以", "因此", "因而", "从而", "由于", "导致",
        "引起", "造成", "为此", "基于",
    ],
    "sequential": [
        "first", "firstly", "second", "secondly", "third", "thirdly",
        "next", "then", "finally", "lastly", "subsequently", "afterwards",
        "meanwhile", "before", "after", "initially", "ultimately",
        "to begin with", "in conclusion", "in summary",
        # 中文
        "首先", "其次", "再次", "最后", "第一", "第二", "第三",
        "然后", "接着", "随后", "同时", "之前", "之后",
        "综上所述", "总之", "概括",
    ],
}


def transition_word_analysis(text: str) -> dict[str, float]:
    """分析文本中的过渡词使用情况。

    人类写作者倾向于使用多样化的过渡词，而AI生成的文本
    可能过度使用某些过渡词或类别分布不均衡。

    Args:
        text: 输入文本。

    Returns:
        dict，包含各类过渡词的数量、比例和多样性得分。
    """
    text_lower = text.lower()
    category_counts: dict[str, int] = {}
    category_diversity: dict[str, float] = {}

    for category, words in TRANSITION_WORDS.items():
        count = 0
        matched = set()
        for w in words:
            # 单词边界匹配
            pattern = re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)
            found = pattern.findall(text_lower)
            count += len(found)
            if found:
                matched.add(w)

            # 中文匹配（直接子串匹配）
            if re.search(r"[一-鿿]", w):
                count += text.count(w)
                matched.add(w)

        category_counts[category] = count
        total_words_in_category = len(TRANSITION_WORDS[category])
        category_diversity[category] = (
            len(matched) / total_words_in_category if total_words_in_category > 0 else 0.0
        )

    total_transitions = sum(category_counts.values())
    return {
        "total_transitions": float(total_transitions),
        "additive_count": float(category_counts["additive"]),
        "adversative_count": float(category_counts["adversative"]),
        "causal_count": float(category_counts["causal"]),
        "sequential_count": float(category_counts["sequential"]),
        "additive_diversity": category_diversity["additive"],
        "adversative_diversity": category_diversity["adversative"],
        "causal_diversity": category_diversity["causal"],
        "sequential_diversity": category_diversity["sequential"],
    }


def coherence_score(text: str) -> float:
    """计算文本的连贯性综合评分。

    基于过渡词的密度和多样性进行加权评分。

    Args:
        text: 输入文本。

    Returns:
        连贯性得分，范围 [0, 1]。
    """
    analysis = transition_word_analysis(text)
    total = analysis["total_transitions"]

    words = len(re.findall(r"\b\w+\b", text))
    if words == 0:
        return 0.0

    # 过渡词密度得分（每100词的过渡词数量，过高或过低都扣分）
    density = total / words * 100
    density_score = 1.0 - abs(density - 5.0) / 15.0  # 最优密度约为5%
    density_score = max(0.0, min(1.0, density_score))

    # 多样性得分：各类过渡词的均匀分布
    category_counts = [
        analysis["additive_count"],
        analysis["adversative_count"],
        analysis["causal_count"],
        analysis["sequential_count"],
    ]
    total_count = sum(category_counts)
    if total_count == 0:
        diversity_score = 0.0
    else:
        proportions = [c / total_count for c in category_counts]
        # 使用熵来衡量分布均匀度，归一化到 [0, 1]
        entropy = -sum(
            p * np.log(p + 1e-10) for p in proportions if p > 0
        )
        max_entropy = np.log(4)
        diversity_score = float(entropy / max_entropy)

    return 0.5 * density_score + 0.5 * diversity_score


# ======================================================================
# 篇章结构分析
# ======================================================================

DISCOURSE_MARKERS: dict[str, list[str]] = {
    "claim": [
        "we argue", "we propose", "we claim", "we believe", "in our view",
        "we hypothesize", "our approach", "this paper proposes",
        "我们提出", "本文提出", "我们认为", "我们主张", "本文认为",
    ],
    "evidence": [
        "we found", "results show", "experiments demonstrate", "data indicate",
        "our analysis reveals", "we observe", "evidence suggests",
        "研究发现", "实验表明", "结果显示", "数据分析表明",
    ],
    "conclusion": [
        "in conclusion", "to summarize", "we conclude", "in summary",
        "overall", "therefore", "thus", "hence",
        "综上所述", "总之", "因此", "由此可知", "我们得出结论",
    ],
}


def discourse_pattern(text: str) -> list[str]:
    """识别文本中的篇章结构模式。

    Args:
        text: 输入文本。

    Returns:
        按顺序排列的篇章段标签列表，如 ['claim', 'evidence', 'conclusion']。
    """
    text_lower = text.lower()
    paragraphs = re.split(r"\n\s*\n", text)
    pattern: list[str] = []

    for para in paragraphs:
        para_lower = para.lower()
        found = False
        for segment_type, markers in DISCOURSE_MARKERS.items():
            if any(m in para_lower for m in markers):
                pattern.append(segment_type)
                found = True
                break
        if not found:
            pattern.append("other")

    return pattern


# ======================================================================
# 基于语义的连贯性分析
# ======================================================================


def local_coherence(sentences: list[str], encoder: SentenceEncoder) -> float:
    """计算局部连贯性——相邻句子间的平均语义相似度。

    人类写作通常具有良好的局部连贯性（相邻句子语义相关），
    AI生成文本可能表现出异常的连贯性模式。

    Args:
        sentences: 句子列表。
        encoder: SentenceEncoder 实例。

    Returns:
        局部连贯性得分，范围 [0, 1]。
    """
    if len(sentences) < 2:
        return 0.0

    vectors = encoder.encode(sentences)
    if len(vectors) < 2:
        return 0.0

    similarities = []
    for i in range(len(vectors) - 1):
        v1, v2 = vectors[i], vectors[i + 1]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            continue
        similarities.append(float(dot / norm))

    if not similarities:
        return 0.0

    return float(np.mean(similarities))


def global_coherence(sentences: list[str], encoder: SentenceEncoder) -> float:
    """计算全局连贯性——每个句子与文档整体语义的相似度。

    衡量每个句子与文档主题的一致性。

    Args:
        sentences: 句子列表。
        encoder: SentenceEncoder 实例。

    Returns:
        全局连贯性得分，范围 [0, 1]。
    """
    if not sentences:
        return 0.0

    vectors = encoder.encode(sentences)
    if len(vectors) == 0:
        return 0.0

    # 文档中心向量
    doc_vector = np.mean(vectors, axis=0)

    # 每句与中心的相似度
    doc_norm = np.linalg.norm(doc_vector)
    if doc_norm == 0:
        return 0.0

    similarities = []
    for v in vectors:
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue
        sim = float(np.dot(v, doc_vector) / (v_norm * doc_norm))
        similarities.append(sim)

    if not similarities:
        return 0.0

    return float(np.mean(similarities))


def coherence_anomaly_score(text: str, encoder: SentenceEncoder) -> float:
    """计算连贯性异常得分，用于AI内容检测。

    结合局部和全局连贯性分析，分数越高越可能是AI生成。

    Args:
        text: 输入文本。
        encoder: SentenceEncoder 实例。

    Returns:
        异常得分，范围 [0, 1]。越高越可能是AI生成。
    """
    sentences = encoder._split_sentences(text)
    if len(sentences) < 3:
        return 0.0

    local = local_coherence(sentences, encoder)
    global_c = global_coherence(sentences, encoder)

    # AI生成文本往往具有异常高的局部连贯性
    # 和异常低或异常高的全局连贯性
    local_anomaly = abs(local - 0.7) / 0.7  # 偏离0.7视为异常
    global_anomaly = abs(global_c - 0.6) / 0.6

    return float(min(1.0, 0.5 * local_anomaly + 0.5 * global_anomaly))


__all__ = [
    "transition_word_analysis",
    "coherence_score",
    "discourse_pattern",
    "local_coherence",
    "global_coherence",
    "coherence_anomaly_score",
]