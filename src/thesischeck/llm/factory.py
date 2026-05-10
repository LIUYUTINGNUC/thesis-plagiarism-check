"""LLM 客户端工厂——根据配置自动创建对应厂商的客户端实例。"""

from __future__ import annotations

from typing import Optional

from thesischeck.llm.base import LLMClient, LLMConfig
from thesischeck.llm.claude import ClaudeClient
from thesischeck.llm.openai_style import (
    PROVIDER_CONFIGS,
    OpenAICompatibleClient,
    OpenAIClient,
)

# 厂商名 → 客户端类映射
_PROVIDER_MAP: dict[str, type[LLMClient]] = {
    "claude": ClaudeClient,
    "openai": OpenAIClient,
}


def create_llm_client(
    provider: Optional[str] = None,
    config: Optional[LLMConfig] = None,
) -> LLMClient:
    """创建 LLM 客户端实例。

    自动识别厂商并实例化对应的客户端实现。
    Claude 使用原生 SDK，其余厂商均使用 OpenAI 兼容客户端。

    Args:
        provider: 厂商名称。为 None 时从 LLMConfig 或环境变量读取。
        config: LLM 配置。为 None 时从环境变量自动加载。

    Returns:
        对应厂商的 LLMClient 实例。

    Raises:
        ValueError: 不支持的厂商或 API key 未配置。

    Examples:
        >>> client = create_llm_client("claude")          # Claude
        >>> client = create_llm_client("deepseek")         # DeepSeek
        >>> client = create_llm_client("qwen")             # 通义千问
    """
    if config is None:
        config = LLMConfig.from_env(provider)

    provider_name = config.provider.lower()

    # 特殊处理: Claude 用原生 SDK
    if provider_name == "claude":
        return ClaudeClient(config)

    # 特殊处理: OpenAI
    if provider_name == "openai":
        return OpenAIClient(config)

    # 检查是否在已知厂商列表中
    if provider_name not in PROVIDER_CONFIGS and provider_name not in _PROVIDER_MAP:
        known = ", ".join(list(_PROVIDER_MAP.keys()) + list(PROVIDER_CONFIGS.keys()))
        raise ValueError(
            f"不支持的 LLM 厂商: '{provider_name}'。"
            f"支持的厂商: {known}"
        )

    return OpenAICompatibleClient(config)


def list_available_providers() -> list[dict[str, str]]:
    """列出所有支持的 LLM 厂商。

    Returns:
        厂商信息列表，每项包含 name 和 description。
    """
    providers = [
        {"name": "claude", "description": "Anthropic Claude（原生API，支持token概率检测）"},
        {"name": "openai", "description": "OpenAI GPT-4o 等"},
    ]

    known_providers = {
        "deepseek": "DeepSeek（深度求索，支持logprobs）",
        "qwen": "阿里通义千问 Qwen",
        "glm": "智谱 GLM",
        "moonshot": "月之暗面 Kimi",
        "baichuan": "百川 Baichuan",
        "yi": "零一万物 Yi",
        "ernie": "百度文心 ERNIE",
        "siliconflow": "SiliconFlow（硅基流动，汇聚多种开源模型）",
        "spark": "讯飞星火",
    }

    for name, desc in known_providers.items():
        providers.append({"name": name, "description": desc})

    return providers


__all__ = ["create_llm_client", "list_available_providers"]