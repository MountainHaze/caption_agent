from __future__ import annotations

import os
from typing import Iterable

from PIL import Image, ImageDraw

from caption_agent.domain.entities.person_instance import PersonInstance


def draw_person_boxes(
    image: Image.Image,
    persons: Iterable[PersonInstance],
    target_id: str,
) -> Image.Image:
    canvas = image.copy()
    drawer = ImageDraw.Draw(canvas)
    for person in persons:
        box = person.bbox
        color = "red" if person.id == target_id else "yellow"
        drawer.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=3)
        drawer.text((box.x1 + 2, box.y1 + 2), person.id, fill=color)
    return canvas


def save_image(image: Image.Image, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    return path

