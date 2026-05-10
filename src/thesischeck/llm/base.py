"""LLM 客户端抽象基类与数据模型。"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Role(Enum):
    """消息角色。"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """对话消息。"""
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass
class LLMConfig:
    """LLM 客户端通用配置。

    各厂商特有的参数通过 extra_kwargs 传递。
    """
    provider: str = "claude"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: float = 60.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, provider: Optional[str] = None) -> "LLMConfig":
        """从环境变量加载配置。

        环境变量前缀规则:
        - LLM_PROVIDER: 厂商
        - {PROVIDER}_API_KEY: API 密钥, 也兼容 LLM_API_KEY
        - {PROVIDER}_BASE_URL: 自定义端点
        - {PROVIDER}_MODEL: 模型名
        """
        provider = provider or os.getenv("LLM_PROVIDER", "claude").lower()
        prefix = provider.upper()

        api_key = (
            os.getenv(f"{prefix}_API_KEY")
            or os.getenv("LLM_API_KEY")
            or ""
        )
        base_url = os.getenv(f"{prefix}_BASE_URL", "")
        model = os.getenv(f"{prefix}_MODEL", "")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class LLMClient(ABC):
    """LLM 客户端抽象基类。

    所有厂商实现需继承此类并实现以下方法。
    统计模块作为降级兜底，当 LLM 不可用时自动切换。
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    # ==================================================================
    # 必须实现的核心接口
    # ==================================================================

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """对话补全。

        Args:
            messages: 对话历史消息列表。
            system: 系统提示词（部分厂商通过参数传入）。
            **kwargs: 厂商特有参数。

        Returns:
            模型回复文本。
        """
        ...

    @abstractmethod
    def chat_with_probs(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> tuple[str, list[dict]]:
        """对话补全并返回 token 级概率。

        用于 AI 生成内容检测——人类写作与 AI 生成的 token
        概率分布存在显著差异。

        Returns:
            (回复文本, token 概率列表 [{"token": str, "logprob": float}, ...])
        """
        ...

    # ==================================================================
    # 可选实现的便捷方法
    # ==================================================================

    def chat_structured(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """结构化输出（默认等价于 chat，子类可重写以支持 JSON mode）。"""
        return self.chat(messages, system=system, **kwargs)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """返回厂商名称标识。"""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """返回当前厂商的默认模型名。"""
        ...

    def count_tokens(self, text: str) -> int:
        """估算 token 数量（默认用简单规则，子类可重写）。"""
        # 粗略估计：英文约 1 token/4 字符，中文约 1 token/2 字符
        char_count = len(text)
        zh_chars = sum(1 for c in text if "一" <= c <= "鿿")
        en_chars = char_count - zh_chars
        return zh_chars // 2 + en_chars // 4 + 1

    def check_available(self) -> bool:
        """检查客户端是否可用（API key 是否配置）。"""
        return bool(self.config.api_key)
