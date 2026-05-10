"""LLM 客户端工厂——根据配置自动创建对应客户端实例。"""

from __future__ import annotations

from typing import Optional

from thesischeck.llm.base import LLMClient, LLMConfig
from thesischeck.llm.claude import ClaudeClient
from thesischeck.llm.openai_style import OpenAIClient, OpenAICompatibleClient


def create_llm_client(
    provider: Optional[str] = None,
    config: Optional[LLMConfig] = None,
) -> LLMClient:
    """创建 LLM 客户端实例。

    Claude 使用原生 SDK，其余任意厂商名均使用 OpenAI 兼容客户端。
    用户通过环境变量配置 base_url 和 model 即可对接任意 API。

    Args:
        provider: 厂商名称。为 None 时从 LLMConfig 或环境变量读取。
        config: LLM 配置。为 None 时从环境变量自动加载。

    Returns:
        对应客户端实例。

    Examples:
        >>> client = create_llm_client("claude")   # Claude 原生
        >>> client = create_llm_client("myapi")    # 任意 OpenAI 兼容 API
    """
    if config is None:
        config = LLMConfig.from_env(provider)

    provider_name = config.provider.lower()

    if provider_name == "claude":
        return ClaudeClient(config)
    if provider_name == "openai":
        return OpenAIClient(config)

    # 其余全部视为 OpenAI 兼容接口
    return OpenAICompatibleClient(config)


def list_available_providers() -> list[dict[str, str]]:
    """列出内置的 LLM 接口类型。

    Returns:
        接口类型列表，每项包含 name 和 description。
    """
    return [
        {"name": "claude", "description": "Anthropic Claude（原生API，支持token概率检测）"},
        {"name": "openai", "description": "OpenAI 兼容接口（可对接任意兼容的 API 服务）"},
    ]


__all__ = ["create_llm_client", "list_available_providers"]
