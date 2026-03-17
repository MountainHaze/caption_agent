from __future__ import annotations

from caption_agent.infrastructure.llm.openai_compatible import (
    build_openai_compatible_client,
)
from caption_agent.infrastructure.llm.multimodal_client import (
    LangChainMultimodalClient,
    MultimodalClientProtocol,
)
from caption_agent.prompts import load_prompt_bundle
from caption_agent.shared.config import AppSettings


def build_multimodal_client(settings: AppSettings) -> MultimodalClientProtocol:
    """
    Factory entry for multimodal client.

    Provider options:
    - mock: local conservative runnable baseline
    - openai: OpenAI official API
    - qwen: DashScope OpenAI-compatible API
    """
    provider = settings.llm_provider.lower()
    prompt_bundle = load_prompt_bundle(settings.prompts_config_path)
    if provider in {"openai", "qwen"}:
        return build_openai_compatible_client(settings, prompt_bundle)
    return LangChainMultimodalClient()
