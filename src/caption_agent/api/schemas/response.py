from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RelationFactResponse(BaseModel):
    relation: str
    confidence: float
    evidence: str | None = None
    target_person_id: str | None = None
    object_type: str | None = None


class CaptionResponse(BaseModel):
    instance_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    person_relations: list[RelationFactResponse] = Field(default_factory=list)
    object_relations: list[RelationFactResponse] = Field(default_factory=list)
    summary: str | None = None
    errors: list[str] = Field(default_factory=list)

