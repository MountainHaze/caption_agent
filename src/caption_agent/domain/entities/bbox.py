from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class BBox(BaseModel):
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")

    @model_validator(mode="after")
    def validate_order(self) -> "BBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must satisfy x2>x1 and y2>y1")
        return self

    @classmethod
    def from_list(cls, values: list[float]) -> "BBox":
        if len(values) != 4:
            raise ValueError("bbox list must have exactly 4 values")
        return cls(x1=values[0], y1=values[1], x2=values[2], y2=values[3])

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

