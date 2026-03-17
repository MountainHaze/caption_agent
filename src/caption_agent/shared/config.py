from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None


@dataclass(frozen=True)
class AppSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    artifact_dir: str = "artifacts"
    person_relation_threshold: float = 0.65
    object_relation_threshold: float = 0.75
    llm_provider: str = "mock"
    llm_model: str = "gpt-4.1-mini"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.1
    prompts_config_path: str = "configs/prompts.yaml"

    @classmethod
    def from_env(cls) -> "AppSettings":
        if load_dotenv is not None:
            load_dotenv()

        provider = os.getenv("CAPTION_AGENT_LLM_PROVIDER", "mock").lower()
        llm_api_key = os.getenv("CAPTION_AGENT_LLM_API_KEY")
        llm_base_url = os.getenv("CAPTION_AGENT_LLM_BASE_URL")
        llm_model = os.getenv("CAPTION_AGENT_LLM_MODEL", "gpt-4.1-mini")

        if provider == "openai":
            llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
            llm_base_url = llm_base_url or os.getenv("OPENAI_BASE_URL")
            llm_model = os.getenv("CAPTION_AGENT_LLM_MODEL", llm_model)

        if provider == "qwen":
            llm_api_key = llm_api_key or os.getenv("DASHSCOPE_API_KEY")
            llm_base_url = llm_base_url or (
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            llm_model = os.getenv("CAPTION_AGENT_LLM_MODEL", "qwen-vl-max")

        return cls(
            host=os.getenv("CAPTION_AGENT_HOST", "127.0.0.1"),
            port=int(os.getenv("CAPTION_AGENT_PORT", "8000")),
            artifact_dir=os.getenv("CAPTION_AGENT_ARTIFACT_DIR", "artifacts"),
            person_relation_threshold=float(
                os.getenv("CAPTION_AGENT_PERSON_REL_THRESHOLD", "0.65")
            ),
            object_relation_threshold=float(
                os.getenv("CAPTION_AGENT_OBJECT_REL_THRESHOLD", "0.75")
            ),
            llm_provider=provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_temperature=float(os.getenv("CAPTION_AGENT_LLM_TEMPERATURE", "0.1")),
            prompts_config_path=os.getenv(
                "CAPTION_AGENT_PROMPTS_CONFIG", "configs/prompts.yaml"
            ),
        )
