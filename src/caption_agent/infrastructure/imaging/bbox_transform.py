from __future__ import annotations

from typing import Literal

from caption_agent.domain.entities.bbox import BBox

BBoxFormat = Literal["auto", "xyxy", "xywh", "norm_xywh"]


def resolve_bbox_to_xyxy(
    raw_bbox: list[float],
    image_width: int,
    image_height: int,
    bbox_format: BBoxFormat = "auto",
) -> BBox:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image size must be positive")

    values = [float(item) for item in raw_bbox]
    if len(values) != 4:
        raise ValueError("bbox must have exactly 4 values")

    inferred_format = _infer_bbox_format(values, bbox_format)
    x1, y1, x2, y2 = _convert_to_xyxy(values, inferred_format, image_width, image_height)
    x1, y1, x2, y2 = _clamp_xyxy(x1, y1, x2, y2, image_width, image_height)
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _infer_bbox_format(values: list[float], bbox_format: BBoxFormat) -> BBoxFormat:
    if bbox_format != "auto":
        return bbox_format

    if all(0.0 <= value <= 1.0 for value in values):
        return "norm_xywh"

    x1, y1, x2, y2 = values
    if x2 > x1 and y2 > y1:
        return "xyxy"
    return "xywh"


def _convert_to_xyxy(
    values: list[float],
    bbox_format: BBoxFormat,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if bbox_format == "xyxy":
        x1, y1, x2, y2 = values[-4:]
        return x1, y1, x2, y2

    if bbox_format == "xywh":
        x, y, w, h = values[-4:]
        return x, y, x + w, y + h

    if bbox_format == "norm_xywh":
        cx, cy, w, h = values
        x1 = (cx - w / 2.0) * image_width
        y1 = (cy - h / 2.0) * image_height
        x2 = (cx + w / 2.0) * image_width
        y2 = (cy + h / 2.0) * image_height
        return x1, y1, x2, y2

    raise ValueError(f"unsupported bbox_format={bbox_format}")


def _clamp_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(image_width - 1), x1))
    y1 = max(0.0, min(float(image_height - 1), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))

    if x2 <= x1:
        x2 = min(float(image_width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_height), y1 + 1.0)
    return x1, y1, x2, y2
