from __future__ import annotations

import math

from caption_agent.domain.entities.bbox import BBox
from caption_agent.domain.entities.person_instance import PersonInstance


def iou(a: BBox, b: BBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = a.area + b.area - inter_area
    return inter_area / union if union > 0 else 0.0


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = a.center
    bx, by = b.center
    return math.dist((ax, ay), (bx, by))


def normalized_distance(a: BBox, b: BBox, image_w: int, image_h: int) -> float:
    diag = math.dist((0.0, 0.0), (float(image_w), float(image_h)))
    if diag <= 0:
        return 1.0
    return center_distance(a, b) / diag


def top_k_neighbors(
    target: PersonInstance,
    persons: list[PersonInstance],
    image_w: int,
    image_h: int,
    top_k: int = 5,
) -> list[PersonInstance]:
    others = [p for p in persons if p.id != target.id]
    ranked = sorted(
        others,
        key=lambda p: normalized_distance(target.bbox, p.bbox, image_w, image_h),
    )
    return ranked[:top_k]

