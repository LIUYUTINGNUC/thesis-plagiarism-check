"""统计特征提取模块——用于AI生成内容检测。

提取文本的统计特征，包括词汇熵、句长分布、词汇丰富度等，
这些特征在人类写作与AI生成文本之间存在显著差异。
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


# ======================================================================
# 基础统计特征
# ======================================================================


def calculate_entropy(text: str) -> float:
    """计算文本的香农熵（基于字符分布）。

    人类写作通常具有较高的熵值（更丰富的字符使用模式），
    而AI生成文本的熵值可能呈现异常分布。

    Args:
        text: 输入文本。

    Returns:
        香农熵值（bits），范围 [0, log2(n_chars)]。
    """
    if not text.strip():
        return 0.0

    char_counts = Counter(text)
    total = len(text)
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in char_counts.values()
    )
    return entropy


def sentence_length_stats(text: str) -> dict[str, float]:
    """计算句子长度分布的统计量。

    AI生成文本往往表现出句子长度分布均匀、方差较小的特点，
    而人类写作的句子长度波动更大。

    Args:
        text: 输入文本。

    Returns:
        dict，包含 mean, std, min, max, q25, q50, q75。
    """
    sentences = re.split(r"[。！？.!?\n]+", text)
    lengths = np.array(
        [len(s.strip().split()) for s in sentences if s.strip()],
        dtype=np.float64,
    )

    if len(lengths) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "q25": 0.0, "q50": 0.0, "q75": 0.0}

    return {
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": float(np.min(lengths)),
        "max": float(np.max(lengths)),
        "q25": float(np.percentile(lengths, 25)),
        "q50": float(np.median(lengths)),
        "q75": float(np.percentile(lengths, 75)),
    }


def vocabulary_richness(text: str) -> float:
    """计算词汇丰富度（型例比 Type-Token Ratio, TTR）。

    更高的 TTR 通常表示更丰富的词汇使用。AI生成文本的
    TTR 可能偏低（倾向于重复使用常见词汇）或异常偏高。

    Args:
        text: 输入文本。

    Returns:
        TTR 值，范围 [0, 1]。
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0

    return len(set(words)) / len(words)


def punctuation_density(text: str) -> float:
    """计算标点符号密度。

    人类写作往往使用更多样化的标点符号，尤其是分号、破折号等，
    而AI生成文本的标点使用可能更为规律。

    Args:
        text: 输入文本。

    Returns:
        标点符号占总字符数的比例。
    """
    if not text.strip():
        return 0.0

    punctuation_chars = set("，。！？；：、""''（）【】《》—…·,.;:!?\"'()[]{}<>-")
    punct_count = sum(1 for c in text if c in punctuation_chars)
    return punct_count / len(text)


_FUNCTION_WORDS_ZH: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "它", "们", "那", "为", "以", "从", "与", "而", "但", "或",
    "被", "把", "对", "等", "之", "所", "能", "可以", "应该", "必须",
    "将", "已", "还", "又", "再", "才", "就", "都", "只", "也",
}

_FUNCTION_WORDS_EN: set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "not", "no", "nor", "so", "if",
    "then", "than", "that", "this", "these", "those", "it", "its",
    "we", "they", "he", "she", "which", "who", "whom", "what",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "some", "any", "such", "only", "own",
    "same", "very", "just", "also", "too", "quite", "rather",
}


def function_word_ratio(text: str) -> float:
    """计算功能词（虚词）占比。

    功能词的使用模式在人类与AI写作之间存在差异。
    AI倾向于过度使用或不当使用某些功能词。

    Args:
        text: 输入文本（中英文混合）。

    Returns:
        功能词占总词数的比例，范围 [0, 1]。
    """
    # 中文分词：按字符分割
    zh_chars = re.findall(r"[一-鿿]", text)
    zh_functional = sum(1 for c in zh_chars if c in _FUNCTION_WORDS_ZH)

    # 英文分词
    en_words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    en_functional = sum(1 for w in en_words if w in _FUNCTION_WORDS_EN)

    total_words = len(zh_chars) + len(en_words)
    if total_words == 0:
        return 0.0

    return (zh_functional + en_functional) / total_words


# ======================================================================
# 可读性评分（近似）
# ======================================================================


def _count_syllables_en(word: str) -> int:
    """粗略估计英文单词的音节数。"""
    word = word.lower()
    syllable_count = 0
    vowels = "aeiouy"
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            syllable_count += 1
        prev_is_vowel = is_vowel
    if word.endswith("e"):
        syllable_count = max(1, syllable_count - 1)
    return max(1, syllable_count)


def readability_scores(text: str) -> dict[str, float]:
    """计算文本可读性评分。

    包含 Flesch Reading Ease 和 Flesch-Kincaid Grade Level，
    用于评估文本的复杂度特征。

    Args:
        text: 输入文本。

    Returns:
        dict，包含 flesch_reading_ease 和 fk_grade_level。
    """
    sentences = re.split(r"[.!?\n]+", text)
    sentence_count = max(len([s for s in sentences if s.strip()]), 1)

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    word_count = len(words)

    if word_count == 0:
        return {"flesch_reading_ease": 0.0, "fk_grade_level": 0.0}

    syllables = sum(_count_syllables_en(w) for w in words)

    # Flesch Reading Ease
    flesch = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllables / word_count)

    # Flesch-Kincaid Grade Level
    fk = 0.39 * (word_count / sentence_count) + 11.8 * (syllables / word_count) - 15.59

    return {
        "flesch_reading_ease": round(max(0, min(100, flesch)), 2),
        "fk_grade_level": round(max(0, fk), 2),
    }


# ======================================================================
# 综合特征提取
# ======================================================================


def extract_all_features(text: str) -> dict[str, float]:
    """提取文本的所有统计特征.

    Args:
        text: 输入文本。

    Returns:
        dict，包含所有特征名称到值的映射。
    """
    features: dict[str, float] = {}

    features["entropy"] = calculate_entropy(text)
    features["vocabulary_richness"] = vocabulary_richness(text)
    features["punctuation_density"] = punctuation_density(text)
    features["function_word_ratio"] = function_word_ratio(text)

    sent_stats = sentence_length_stats(text)
    for key, val in sent_stats.items():
        features[f"sentence_length_{key}"] = val

    readability = readability_scores(text)
    features.update(readability)

    return features


def feature_vector(text: str) -> np.ndarray:
    """将所有特征归一化为向量，用于下游分类。

    Args:
        text: 输入文本。

    Returns:
        numpy 数组，包含归一化后的特征值。
    """
    features = extract_all_features(text)
    # 排除非数值特征（如果有的话），取确定顺序的值
    keys = sorted(features.keys())
    values = np.array([features[k] for k in keys], dtype=np.float32)

    # 简单归一化到 [0, 1]
    v_min, v_max = values.min(), values.max()
    if v_max - v_min > 1e-8:
        values = (values - v_min) / (v_max - v_min)

    return values


__all__ = [
    "calculate_entropy",
    "sentence_length_stats",
    "vocabulary_richness",
    "punctuation_density",
    "function_word_ratio",
    "readability_scores",
    "extract_all_features",
    "feature_vector",
]