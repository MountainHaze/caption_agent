from __future__ import annotations

import os

from caption_agent.graph.state import CaptionState
from caption_agent.infrastructure.imaging.annotator import draw_person_boxes, save_image
from caption_agent.infrastructure.imaging.cropper import (
    crop_with_bbox,
    expand_bbox,
    load_image,
)
from caption_agent.infrastructure.imaging.geometry import top_k_neighbors


def prepare_visual_context_node(state: CaptionState) -> dict:
    errors = list(state.get("errors", []))
    target = state["target_instance"]
    persons = state["instances"]
    request_id = state.get("request_id", "unknown")
    artifact_root = state.get("artifact_dir", "artifacts")
    image_ref = state["image_ref"]

    try:
        image = load_image(image_ref)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"image_load_failed: {exc}")
        return {
            "errors": errors,
            "image_width": 1,
            "image_height": 1,
            "neighbor_persons": [],
            "image_context": {},
        }

    image_w, image_h = image.size
    neighbors = top_k_neighbors(target, persons, image_w, image_h, top_k=5)

    req_dir = os.path.join(artifact_root, request_id)
    os.makedirs(req_dir, exist_ok=True)

    annotated = draw_person_boxes(image, persons, target.id)
    annotated_path = save_image(annotated, os.path.join(req_dir, "annotated.png"))

    tight_crop = crop_with_bbox(image, target.bbox)
    tight_crop_path = save_image(tight_crop, os.path.join(req_dir, "target_tight.png"))

    context_box = expand_bbox(target.bbox, image_w, image_h, ratio=0.35)
    context_crop = crop_with_bbox(image, context_box)
    context_crop_path = save_image(context_crop, os.path.join(req_dir, "target_context.png"))

    return {
        "errors": errors,
        "image_width": image_w,
        "image_height": image_h,
        "neighbor_persons": neighbors,
        "image_context": {
            "image_width": image_w,
            "image_height": image_h,
            "annotated_image": annotated_path,
            "target_tight_crop": tight_crop_path,
            "target_context_crop": context_crop_path,
        },
    }

