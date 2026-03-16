from caption_agent.infrastructure.imaging.bbox_transform import resolve_bbox_to_xyxy


def test_resolve_norm_xywh_to_xyxy() -> None:
    bbox = resolve_bbox_to_xyxy(
        raw_bbox=[0.620039, 0.5939, 0.172415, 0.14608],
        image_width=768,
        image_height=768,
        bbox_format="norm_xywh",
    )
    assert bbox.x1 < bbox.x2
    assert bbox.y1 < bbox.y2
    assert 0 <= bbox.x1 <= 768
    assert 0 <= bbox.y1 <= 768


def test_reject_five_values_bbox() -> None:
    try:
        resolve_bbox_to_xyxy(
            raw_bbox=[0, 0.620039, 0.5939, 0.172415, 0.14608],
            image_width=768,
            image_height=768,
            bbox_format="auto",
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "exactly 4 values" in str(exc)
