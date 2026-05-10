"""GPT指纹检测模块——用于AI生成内容识别。

通过分析文本的统计特征如突发性（burstiness）、困惑度（perplexity）、
重复模式等，结合多指标加权评分，判断文本是否由AI生成。
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# ======================================================================
# 突发性分析
# ======================================================================


def burstiness_score(text: str) -> float:
    """计算文本的突发性得分——衡量句子长度方差。

    人类写作通常表现出高突发性（句子长度变化大），
    AI生成文本往往长度分布更均匀（低突发性）。

    Args:
        text: 输入文本。

    Returns:
        突发性得分，范围 [0, 1]。越高越像人类写作。
    """
    sentences = re.split(r"[。！？.!?\n]+", text)
    lengths = [len(s.strip().split()) for s in sentences if s.strip()]

    if len(lengths) < 2:
        return 0.0

    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 0.0

    std_len = np.std(lengths)
    # 变异系数（CV）作为突发性指标，归一化到 [0, 1]
    cv = std_len / mean_len
    # 典型人类文本的CV约为0.5-1.0，AI文本约为0.2-0.5
    score = min(1.0, cv / 1.0)
    return float(score)


# ======================================================================
# 困惑度计算
# ======================================================================


def perplexity_score(text: str, model_name: str = "gpt2") -> float:
    """计算文本的困惑度。

    使用 HuggingFace 的预训练语言模型计算困惑度。
    AI生成文本通常具有较低的困惑度（更" predictable"）。

    Args:
        text: 输入文本。
        model_name: HuggingFace 模型名称，默认为 'gpt2'。

    Returns:
        困惑度值。越高表示文本越"出乎意料"（更像人类）。
    """
    if not text.strip():
        return 0.0

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        return float(math.exp(loss))
    except (ImportError, Exception):
        # 离线或模型不可用时，使用近似方法
        return _approximate_perplexity(text)


def _approximate_perplexity(text: str) -> float:
    """离线近似困惑度：基于n-gram频率估算。

    当无法加载预训练模型时使用此方法。

    Args:
        text: 输入文本。

    Returns:
        近似困惑度值。
    """
    import torch  # noqa: F401

    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 2:
        return 50.0

    # 使用bigram频率估算
    bigrams = zip(words[:-1], words[1:])
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(words)

    total_prob = 0.0
    count = 0
    for (w1, w2), bg_count in bigram_counts.items():
        if unigram_counts[w1] > 0:
            # P(w2|w1) smoothed
            prob = (bg_count + 1) / (unigram_counts[w1] + len(unigram_counts))
            total_prob += math.log2(prob + 1e-10)
            count += bg_count

    if count == 0:
        return 50.0

    avg_log_prob = total_prob / count
    perplexity = 2 ** (-avg_log_prob)
    return min(1000.0, max(1.0, perplexity))


# ======================================================================
# 重复模式分析
# ======================================================================


def repetition_analysis(text: str) -> dict[str, float]:
    """分析文本中的n-gram重复率。

    AI生成文本往往表现出异常的重复模式：过多或过少。

    Args:
        text: 输入文本。

    Returns:
        dict，包含各n-gram级别的重复率。
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 3:
        return {"unigram_repetition": 0.0, "bigram_repetition": 0.0,
                "trigram_repetition": 0.0}

    def _repetition_rate(tokens: list[str], n: int) -> float:
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        unique = set(ngrams)
        return 1.0 - (len(unique) / len(ngrams))

    return {
        "unigram_repetition": _repetition_rate(words, 1),
        "bigram_repetition": _repetition_rate(words, 2),
        "trigram_repetition": _repetition_rate(words, 3),
    }


# ======================================================================
# Token概率分布分析
# ======================================================================


def token_probability_analysis(text: str, model=None) -> dict[str, float]:
    """分析文本的token级概率分布特征。

    Args:
        text: 输入文本。
        model: 可选的语言模型（预留接口）。

    Returns:
        dict，包含概率分布统计特征（均值、方差、偏度等）。
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 3:
        return {"mean_prob": 0.0, "prob_variance": 0.0,
                "prob_skewness": 0.0, "high_freq_ratio": 0.0}

    counts = Counter(words)
    total = len(words)

    freq = np.array([counts[w] for w in words], dtype=np.float32)
    prob = freq / total

    return {
        "mean_prob": float(np.mean(prob)),
        "prob_variance": float(np.var(prob)),
        "prob_skewness": float(np.nan_to_num(np.mean(((prob - np.mean(prob)) / (np.std(prob) + 1e-10)) ** 3), nan=0.0)),
        "high_freq_ratio": float(np.sum(prob > 0.05) / len(prob)),
    }


# ======================================================================
# 综合评分
# ======================================================================


def fingerprint_vector(text: str, model=None) -> np.ndarray:
    """计算所有指纹特征合并为向量。

    Args:
        text: 输入文本。
        model: 可选的语言模型。

    Returns:
        numpy 数组，包含所有指纹特征。
    """
    burstiness = burstiness_score(text)
    repetition = repetition_analysis(text)
    token_prob = token_probability_analysis(text)

    features = [
        burstiness,
        repetition["unigram_repetition"],
        repetition["bigram_repetition"],
        repetition["trigram_repetition"],
        token_prob["mean_prob"],
        token_prob["prob_variance"],
        token_prob["prob_skewness"],
        token_prob["high_freq_ratio"],
    ]

    return np.array(features, dtype=np.float32)


def combined_ai_score(text: str, model=None) -> float:
    """计算综合AI生成概率得分。

    基于突发性、重复率、概率分布等多个指标进行加权评分。

    Args:
        text: 输入文本。

    Returns:
        AI生成概率得分，范围 [0, 1]。
        0 = 极可能人类写作，1 = 极可能AI生成。
    """
    if not text.strip():
        return 0.0

    burstiness = burstiness_score(text)
    # 突发性越低越可能是AI生成，所以取补
    burstiness_ai_score = 1.0 - burstiness

    repetition = repetition_analysis(text)
    # 重复率异常（过高或过低）都可能是AI
    unigram_rep = repetition["unigram_repetition"]
    bigram_rep = repetition["bigram_repetition"]
    rep_anomaly = abs(unigram_rep - 0.3) / 0.3
    bigram_anomaly = abs(bigram_rep - 0.1) / 0.1
    rep_ai_score = min(1.0, 0.5 * rep_anomaly + 0.5 * bigram_anomaly)

    token_prob = token_probability_analysis(text)
    # 概率方差低 + 高频词比例高 → 可能AI
    prob_ai_score = min(1.0, (
        0.4 * (1.0 - min(1.0, token_prob["prob_variance"] * 10)) +
        0.6 * token_prob["high_freq_ratio"]
    ))

    # 加权综合
    weights = [0.35, 0.35, 0.30]
    scores = [burstiness_ai_score, rep_ai_score, prob_ai_score]
    total = sum(w * s for w, s in zip(weights, scores))

    return float(min(1.0, max(0.0, total)))


def ai_detection_report(text: str) -> dict:
    """生成完整的AI检测报告。

    Args:
        text: 输入文本。

    Returns:
        dict，包含各个维度的评分和最终判定。
    """
    burstiness = burstiness_score(text)
    repetition = repetition_analysis(text)
    token_prob = token_probability_analysis(text)
    combined = combined_ai_score(text)

    # 判定规则
    if combined >= 0.7:
        verdict = "likely_ai_generated"
        confidence = "high"
    elif combined >= 0.5:
        verdict = "possibly_ai_generated"
        confidence = "medium"
    elif combined >= 0.3:
        verdict = "possibly_human_written"
        confidence = "medium"
    else:
        verdict = "likely_human_written"
        confidence = "high"

    return {
        "overall_ai_score": round(combined, 4),
        "verdict": verdict,
        "confidence": confidence,
        "details": {
            "burstiness": round(burstiness, 4),
            "burstiness_interpretation": (
                "high_variance" if burstiness > 0.5 else "low_variance"
            ),
            "unigram_repetition": round(repetition["unigram_repetition"], 4),
            "bigram_repetition": round(repetition["bigram_repetition"], 4),
            "trigram_repetition": round(repetition["trigram_repetition"], 4),
            "mean_token_probability": round(token_prob["mean_prob"], 4),
            "prob_variance": round(token_prob["prob_variance"], 6),
        },
    }


__all__ = [
    "burstiness_score",
    "perplexity_score",
    "repetition_analysis",
    "token_probability_analysis",
    "fingerprint_vector",
    "combined_ai_score",
    "ai_detection_report",
]
