from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RelationFact(BaseModel):
    relation: str = Field(..., description="Controlled relation label")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: Optional[str] = Field(default=None, description="Short rationale")
    target_person_id: Optional[str] = Field(
        default=None, description="Used for person relations"
    )
    object_type: Optional[str] = Field(default=None, description="Used for object relations")

