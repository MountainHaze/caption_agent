from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from caption_agent.infrastructure.llm.multimodal_client import LangChainMultimodalClient
from caption_agent.prompts import PromptBundle
from caption_agent.shared.config import AppSettings


def build_openai_compatible_client(
    settings: AppSettings,
    prompts: PromptBundle,
) -> LangChainMultimodalClient:
    if not settings.llm_api_key:
        raise ValueError(
            "Missing API key. Set CAPTION_AGENT_LLM_API_KEY or provider-specific key."
        )

    chat_model = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
    )

    return LangChainMultimodalClient(
        attribute_runnable=RunnableLambda(
            lambda payload: _extract_attributes(chat_model, prompts, payload)
        ),
        person_relation_runnable=RunnableLambda(
            lambda payload: _extract_person_relations(chat_model, prompts, payload)
        ),
        object_relation_runnable=RunnableLambda(
            lambda payload: _extract_object_relations(chat_model, prompts, payload)
        ),
        summary_runnable=RunnableLambda(
            lambda payload: _compose_summary(chat_model, prompts, payload)
        ),
    )


def _extract_attributes(
    model: ChatOpenAI,
    prompts: PromptBundle,
    payload: dict[str, Any],
) -> dict[str, Any]:
    target = payload["target"]
    image_context = payload["image_context"]
    language = payload.get("language", "zh")

    user_text = (
        "Target person attributes extraction.\n"
        f"target_id: {target.id}\n"
        f"target_bbox: {target.bbox.to_list()}\n"
        f"language: {language}\n"
        "Return strict JSON object with keys:\n"
        "gender_presentation, age_group, visibility, clothing, appearance.\n"
        "Use uncertain when not confident."
    )
    images = [
        image_context.get("annotated_image"),
        image_context.get("target_tight_crop"),
        image_context.get("target_context_crop"),
    ]
    default_payload = {
        "gender_presentation": {"value": "uncertain", "confidence": 0.3},
        "age_group": {"value": "uncertain", "confidence": 0.3},
        "visibility": {
            "full_body": False,
            "face_visible": False,
            "occlusion_level": "medium",
        },
        "clothing": {
            "upper_garment": "unknown",
            "lower_garment": "unknown",
            "outerwear": None,
            "footwear": None,
            "accessories": [],
        },
        "appearance": {
            "hair": None,
            "activity": None,
            "orientation": "unknown",
        },
    }
    return _invoke_json(model, prompts.attribute, user_text, images, default_payload)


def _extract_person_relations(
    model: ChatOpenAI,
    prompts: PromptBundle,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    target = payload["target"]
    neighbors = payload.get("neighbors", [])
    image_context = payload["image_context"]

    neighbors_text = [
        f"id={person.id}, bbox={person.bbox.to_list()}" for person in neighbors
    ]
    user_text = (
        "Target person to person relation extraction.\n"
        f"target_id: {target.id}\n"
        f"target_bbox: {target.bbox.to_list()}\n"
        "neighbors:\n"
        + "\n".join(neighbors_text)
        + "\nReturn JSON array. Each item keys: relation, target_person_id, confidence, evidence.\n"
        "Allowed relation values: next_to_person, in_front_of_person, behind_person, grouped_with, facing_person, walking_with."
    )
    images = [image_context.get("annotated_image"), image_context.get("target_context_crop")]
    return _invoke_json(model, prompts.person_relation, user_text, images, [])


def _extract_object_relations(
    model: ChatOpenAI,
    prompts: PromptBundle,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    target = payload["target"]
    image_context = payload["image_context"]

    user_text = (
        "Target person to object weak relation extraction.\n"
        f"target_id: {target.id}\n"
        f"target_bbox: {target.bbox.to_list()}\n"
        "Return JSON array only when clear evidence exists.\n"
        "Each item keys: relation, object_type, confidence, evidence.\n"
        "Allowed relation values: holding_object, carrying_object, near_vehicle, sitting_on_object, using_object."
    )
    images = [image_context.get("annotated_image"), image_context.get("target_context_crop")]
    return _invoke_json(model, prompts.object_relation, user_text, images, [])


def _compose_summary(
    model: ChatOpenAI,
    prompts: PromptBundle,
    payload: dict[str, Any],
) -> str:
    attributes = payload["attributes"]
    person_relations = payload["person_relations"]
    object_relations = payload["object_relations"]
    language = payload.get("language", "zh")

    user_text = (
        "Compose short factual summary from structured data only.\n"
        f"language: {language}\n"
        f"attributes: {json.dumps(attributes, ensure_ascii=True)}\n"
        f"person_relations: {json.dumps(person_relations, ensure_ascii=True)}\n"
        f"object_relations: {json.dumps(object_relations, ensure_ascii=True)}\n"
        "Do not add facts not present in the data."
    )
    response = model.invoke(
        [
            SystemMessage(content=prompts.summary),
            HumanMessage(content=user_text),
        ]
    )
    return str(response.content).strip()


def _invoke_json(
    model: ChatOpenAI,
    system_prompt: str,
    user_text: str,
    image_paths: list[str | None],
    default_payload: Any,
) -> Any:
    content_blocks: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for path in image_paths:
        if not path:
            continue
        data_url = _to_data_url(path)
        if not data_url:
            continue
        content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})

    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content_blocks),
        ]
    )
    return _parse_json_from_text(str(response.content), default_payload)


def _to_data_url(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    mime_type, _ = mimetypes.guess_type(file_path.as_posix())
    mime_type = mime_type or "image/png"
    image_bytes = file_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _parse_json_from_text(text: str, default_payload: Any) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            if lines[-1].startswith("```"):
                lines = lines[1:-1]
            else:
                lines = lines[1:]
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except Exception:  # noqa: BLE001
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:  # noqa: BLE001
                pass
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:  # noqa: BLE001
                pass
    return default_payload
