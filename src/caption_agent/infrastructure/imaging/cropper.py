from __future__ import annotations

import base64
import io
import os

import httpx
from PIL import Image

from caption_agent.domain.entities.bbox import BBox


def load_image(image_ref: str) -> Image.Image:
    """Load image from data URL, http(s) URL, or local path."""
    if image_ref.startswith("data:image"):
        _, encoded = image_ref.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        with httpx.Client(timeout=20.0) as client:
            response = client.get(image_ref)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")

    if os.path.exists(image_ref):
        return Image.open(image_ref).convert("RGB")

    raise FileNotFoundError(
        "image is not a valid local path or data URL for this MVP implementation"
    )


def crop_with_bbox(image: Image.Image, bbox: BBox) -> Image.Image:
    return image.crop((int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)))


def expand_bbox(bbox: BBox, width: int, height: int, ratio: float = 0.25) -> BBox:
    x_margin = bbox.width * ratio
    y_margin = bbox.height * ratio
    x1 = max(0.0, bbox.x1 - x_margin)
    y1 = max(0.0, bbox.y1 - y_margin)
    x2 = min(float(width), bbox.x2 + x_margin)
    y2 = min(float(height), bbox.y2 + y_margin)
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)
