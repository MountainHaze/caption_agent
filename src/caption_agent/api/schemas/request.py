from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class PersonInstanceInput(BaseModel):
    id: str = Field(..., min_length=1)
    bbox: list[float] | str = Field(..., description="Supports [x1,y1,x2,y2], [x,y,w,h], or YOLO normalized format.")
    bbox_format: Literal["auto", "xyxy", "xywh", "norm_xywh"] = "auto"
    score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float] | str) -> list[float]:
        if isinstance(value, str):
            parts = [item for item in re.split(r"[,\s]+", value.strip()) if item]
            value = [float(item) for item in parts]
        if len(value) != 4:
            raise ValueError("bbox must contain exactly 4 values")
        return [float(item) for item in value]


class CaptionRequest(BaseModel):
    image: str | None = Field(
        default=None,
        description="Image source. Supports data URL, http(s) URL, or local debug path.",
    )
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image bytes (preferred for user-facing API).",
    )
    image_mime_type: str = Field(default="image/jpeg")
    instances: list[PersonInstanceInput] = Field(default_factory=list)
    target_instance_id: str = Field(..., min_length=1)
    language: str = Field(default="zh")
    include_summary: bool = Field(default=True)

    @field_validator("instances")
    @classmethod
    def validate_instances(cls, value: list[PersonInstanceInput]) -> list[PersonInstanceInput]:
        if not value:
            raise ValueError("instances cannot be empty")
        ids = [item.id for item in value]
        if len(set(ids)) != len(ids):
            raise ValueError("instance id must be unique")
        return value

    @field_validator("image_mime_type")
    @classmethod
    def validate_image_mime_type(cls, value: str) -> str:
        return value.strip() or "image/jpeg"

    @field_validator("image_base64", "image")
    @classmethod
    def normalize_empty_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("target_instance_id")
    @classmethod
    def validate_target_instance_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("target_instance_id cannot be empty")
        return stripped

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str) -> str:
        stripped = value.strip()
        return stripped or "zh"

    @property
    def has_image_input(self) -> bool:
        return bool(self.image or self.image_base64)

    @model_validator(mode="after")
    def validate_image_input(self) -> "CaptionRequest":
        if not self.image and not self.image_base64:
            raise ValueError("Provide either image or image_base64")
        instance_ids = {item.id for item in self.instances}
        if self.target_instance_id not in instance_ids:
            raise ValueError("target_instance_id must exist in instances")
        return self
