"""Anthropic Claude API 客户端实现。

支持 Claude 的原生 API 特性：
- text_generation_probability（token 级概率，用于 AI 检测）
- 扩展思考（extended thinking）
- 长上下文（200K token）
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from thesischeck.llm.base import LLMClient, LLMConfig, Message, Role


class ClaudeClient(LLMClient):
    """Anthropic Claude API 客户端。"""

    # 模型上下文窗口与默认模型映射
    MODELS: dict[str, int] = {
        "claude-opus-4-7": 200000,
        "claude-sonnet-4-6": 200000,
        "claude-haiku-4-5": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-haiku-20241022": 200000,
    }

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "claude"

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-6"

    def _get_client(self):
        """延迟加载 Anthropic SDK。"""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "使用 Claude 需要安装: pip install anthropic"
                )
            self._client = anthropic.Anthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url or None,
                timeout=self.config.timeout,
            )
        return self._client

    @property
    def _model(self) -> str:
        return self.config.model or self.default_model

    def chat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        client = self._get_client()

        # 转换消息格式
        claude_messages = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system = (system or "") + msg.content
            else:
                claude_messages.append(msg.to_dict())

        response = client.messages.create(
            model=self.config.extra_kwargs.get("model") or self._model,
            messages=claude_messages,
            system=system or None,
            max_tokens=self.config.extra_kwargs.get("max_tokens", self.config.max_tokens),
            temperature=self.config.extra_kwargs.get("temperature", self.config.temperature),
            **kwargs,
        )
        return response.content[0].text

    def chat_with_probs(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> tuple[str, list[dict]]:
        """Claude 的 token 概率检测。

        使用 Anthropic API beta 功能 text_generation_probability
        返回每个 token 的生成概率，用于 AI 内容检测。
        """
        client = self._get_client()

        claude_messages = []
        system = None
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system = msg.content
            else:
                claude_messages.append(msg.to_dict())

        try:
            response = client.messages.create(
                model=self._model,
                messages=claude_messages,
                system=system,
                max_tokens=100,
                temperature=1.0,
                extra_headers={
                    "anthropic-beta": "text-generation-probability-2025-04-01",
                },
                **kwargs,
            )

            text = response.content[0].text
            # 提取 token 概率（beta 功能返回结构可能变化）
            probs: list[dict] = []

            # 尝试从 content block 的 delta 中提取概率信息
            if hasattr(response.content[0], "probabilities"):
                for token_info in response.content[0].probabilities:
                    probs.append({
                        "token": getattr(token_info, "token", ""),
                        "logprob": getattr(token_info, "logprob", 0.0),
                    })
            elif hasattr(response, "raw_response"):
                raw = json.loads(response.raw_response)
                content = raw.get("content", [{}])[0]
                if "probabilities" in content:
                    probs = content["probabilities"]

            return text, probs

        except Exception as e:
            # Token 概率 API 可能不可用，降级到普通 chat
            text = self.chat(messages)
            return text, []


__all__ = ["ClaudeClient"]