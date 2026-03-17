from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ConfidenceLabel(BaseModel):
    value: str = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisibilityAttrs(BaseModel):
    full_body: bool = False
    face_visible: bool = False
    occlusion_level: Literal["none", "low", "medium", "high"] = "medium"


class ClothingAttrs(BaseModel):
    upper_garment: str | None = None
    lower_garment: str | None = None
    outerwear: str | None = None
    footwear: str | None = None
    accessories: list[str] = Field(default_factory=list)


class AppearanceAttrs(BaseModel):
    hair: str | None = None
    activity: str | None = None
    orientation: Literal["front-facing", "back-facing", "side-facing", "unknown"] = (
        "unknown"
    )


class AttributesPayload(BaseModel):
    gender_presentation: ConfidenceLabel = Field(default_factory=ConfidenceLabel)
    age_group: ConfidenceLabel = Field(default_factory=ConfidenceLabel)
    visibility: VisibilityAttrs = Field(default_factory=VisibilityAttrs)
    clothing: ClothingAttrs = Field(default_factory=ClothingAttrs)
    appearance: AppearanceAttrs = Field(default_factory=AppearanceAttrs)


class RelationPayload(BaseModel):
    relation: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str | None = None
    target_person_id: str | None = None
    object_type: str | None = None

    @field_validator("target_person_id", mode="before")
    @classmethod
    def normalize_target_person_id(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return str(int(value))
        if isinstance(value, str):
            return value.strip() or None
        return str(value)
