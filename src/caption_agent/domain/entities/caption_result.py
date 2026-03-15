from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .relation_result import RelationFact


class CaptionResult(BaseModel):
    instance_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    person_relations: list[RelationFact] = Field(default_factory=list)
    object_relations: list[RelationFact] = Field(default_factory=list)
    summary: str | None = None

