from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from caption_agent.prompts.attribute_prompt import ATTRIBUTE_PROMPT
from caption_agent.prompts.object_relation_prompt import OBJECT_RELATION_PROMPT
from caption_agent.prompts.person_relation_prompt import PERSON_RELATION_PROMPT
from caption_agent.prompts.summary_prompt import SUMMARY_PROMPT


@dataclass(frozen=True)
class PromptBundle:
    attribute: str
    person_relation: str
    object_relation: str
    summary: str


def default_prompt_bundle() -> PromptBundle:
    return PromptBundle(
        attribute=ATTRIBUTE_PROMPT.strip(),
        person_relation=PERSON_RELATION_PROMPT.strip(),
        object_relation=OBJECT_RELATION_PROMPT.strip(),
        summary=SUMMARY_PROMPT.strip(),
    )


def load_prompt_bundle(config_path: str | None) -> PromptBundle:
    default_bundle = default_prompt_bundle()
    if not config_path:
        return default_bundle

    file_path = Path(config_path)
    if not file_path.exists():
        return default_bundle

    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return default_bundle

    prompts_data: dict[str, Any]
    if isinstance(raw, dict) and isinstance(raw.get("prompts"), dict):
        prompts_data = raw["prompts"]
    elif isinstance(raw, dict):
        prompts_data = raw
    else:
        prompts_data = {}

    return PromptBundle(
        attribute=_pick_prompt(prompts_data, "attribute", default_bundle.attribute),
        person_relation=_pick_prompt(
            prompts_data, "person_relation", default_bundle.person_relation
        ),
        object_relation=_pick_prompt(
            prompts_data, "object_relation", default_bundle.object_relation
        ),
        summary=_pick_prompt(prompts_data, "summary", default_bundle.summary),
    )


def _pick_prompt(data: dict[str, Any], key: str, default: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        return default
    cleaned = value.strip()
    return cleaned or default

