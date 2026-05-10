"""LLM 客户端抽象层——统一多厂商大模型API调用。

支持 Claude（原生API）、OpenAI 以及所有兼容 OpenAI 格式的国产模型。
"""

from thesischeck.llm.base import LLMClient, LLMConfig, Message, Role
from thesischeck.llm.claude import ClaudeClient
from thesischeck.llm.openai_style import (
    OpenAIClient,
    OpenAICompatibleClient,
)
from thesischeck.llm.factory import create_llm_client, list_available_providers

__all__ = [
    "LLMClient", "LLMConfig", "Message", "Role",
    "ClaudeClient", "OpenAIClient", "OpenAICompatibleClient",
    "create_llm_client", "list_available_providers",
]