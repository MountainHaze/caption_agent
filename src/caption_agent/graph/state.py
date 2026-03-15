from __future__ import annotations

from typing import Any, TypedDict

from caption_agent.domain.entities.person_instance import PersonInstance


class CaptionState(TypedDict, total=False):
    request_id: str
    image_ref: str
    language: str
    include_summary: bool
    artifact_dir: str

    instances: list[PersonInstance]
    target_instance_id: str
    target_instance: PersonInstance
    neighbor_persons: list[PersonInstance]

    image_width: int
    image_height: int
    image_context: dict[str, Any]

    raw_attributes: dict[str, Any]
    raw_person_relations: list[dict[str, Any]]
    raw_object_relations: list[dict[str, Any]]

    verified_attributes: dict[str, Any]
    verified_person_relations: list[dict[str, Any]]
    verified_object_relations: list[dict[str, Any]]

    summary: str | None
    final_result: dict[str, Any]
    errors: list[str]

