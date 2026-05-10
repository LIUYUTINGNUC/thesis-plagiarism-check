"""OpenAI 风格 API 客户端——支持 OpenAI 及所有兼容的国产模型。

兼容厂商列表（均支持 OpenAI chat completions 格式）：
- OpenAI: GPT-4o, GPT-4o-mini
- DeepSeek: deepseek-chat, deepseek-reasoner
- 阿里通义千问 (Qwen): qwen-turbo, qwen-plus, qwen-max
- 智谱 GLM: glm-4-plus, glm-4-air
- 月之暗面 Moonshot (Kimi): moonshot-v1-8k, moonshot-v1-32k
- 百川 Baichuan: Baichuan4, Baichuan3-Turbo
- 零一万物 Yi: yi-large, yi-medium
- 百度文心 ERNIE: ernie-4.0, ernie-3.5（通过兼容接口）
- SiliconFlow / 讯飞星火 等
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from thesischeck.llm.base import LLMClient, LLMConfig, Message, Role


# ======================================================================
# 厂商默认配置映射
# ======================================================================

PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-plus",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
    },
    "baichuan": {
        "base_url": "https://api.baichuan-ai.com/v1",
        "model": "Baichuan4",
    },
    "yi": {
        "base_url": "https://api.lingyiwanwu.com/v1",
        "model": "yi-large",
    },
    "ernie": {
        "base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat",
        "model": "ernie-4.0",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-72B-Instruct",
    },
    "spark": {
        "base_url": "https://spark-api-open.xf-yun.com/v1",
        "model": "generalv3.5",
    },
}


class OpenAICompatibleClient(LLMClient):
    """兼容 OpenAI chat completions 格式的通用客户端。

    适用于大部分国产大模型，只需切换 base_url 和 model 即可。
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return self.config.provider

    @property
    def default_model(self) -> str:
        provider_cfg = PROVIDER_CONFIGS.get(self.config.provider, {})
        return provider_cfg.get("model", "gpt-4o")

    @property
    def _base_url(self) -> str:
        if self.config.base_url:
            return self.config.base_url
        provider_cfg = PROVIDER_CONFIGS.get(self.config.provider, {})
        return provider_cfg.get("base_url", "https://api.openai.com/v1")

    @property
    def _model(self) -> str:
        return self.config.model or self.default_model

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI as OpenAISDK
            except ImportError:
                raise ImportError("需要安装: pip install openai")
            self._client = OpenAISDK(
                api_key=self.config.api_key,
                base_url=self._base_url,
                timeout=self.config.timeout,
            )
        return self._client

    def chat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        client = self._get_client()
        openai_messages = []

        if system:
            openai_messages.append({"role": "system", "content": system})
        for msg in messages:
            openai_messages.append(msg.to_dict())

        response = client.chat.completions.create(
            model=kwargs.pop("model", self._model),
            messages=openai_messages,
            temperature=kwargs.pop("temperature", self.config.temperature),
            max_tokens=kwargs.pop("max_tokens", self.config.max_tokens),
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def chat_with_probs(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> tuple[str, list[dict]]:
        """获取 token 级 logprobs（需厂商支持）。

        OpenAI / DeepSeek 等支持 logprobs 参数。
        不支持的厂商会降级到普通 chat + 空概率列表。
        """
        client = self._get_client()
        openai_messages = []

        for msg in messages:
            openai_messages.append(msg.to_dict())

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=openai_messages,
                temperature=1.0,
                max_tokens=100,
                logprobs=True,
                top_logprobs=5,
                **kwargs,
            )

            text = response.choices[0].message.content or ""
            logprobs_data = response.choices[0].logprobs
            probs: list[dict] = []

            if logprobs_data and logprobs_data.content:
                for lp in logprobs_data.content:
                    top_lps = lp.top_logprobs or []
                    probs.append({
                        "token": lp.token,
                        "logprob": lp.logprob,
                        "top_logprobs": [
                            {"token": t.token, "logprob": t.logprob}
                            for t in top_lps
                        ],
                    })

            return text, probs

        except Exception:
            text = self.chat(messages)
            return text, []

    def chat_structured(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        response_format: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        """结构化输出（JSON mode）。"""
        client = self._get_client()
        openai_messages = []

        if system:
            openai_messages.append({"role": "system", "content": system})
        for msg in messages:
            openai_messages.append(msg.to_dict())

        body = {
            "model": self._model,
            "messages": openai_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        response = client.chat.completions.create(**body, **kwargs)
        return response.choices[0].message.content or ""


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI 专用客户端（默认配置）。"""

    def __init__(self, config: LLMConfig):
        if not config.base_url:
            config.base_url = "https://api.openai.com/v1"
        if not config.model:
            config.model = "gpt-4o"
        super().__init__(config)

    @property
    def provider_name(self) -> str:
        return "openai"


__all__ = [
    "OpenAIClient",
    "OpenAICompatibleClient",
    "PROVIDER_CONFIGS",
]