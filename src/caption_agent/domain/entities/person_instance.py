from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .bbox import BBox


class PersonInstance(BaseModel):
    id: str = Field(..., description="Unique person instance id")
    bbox: BBox = Field(..., description="Person bounding box")
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

