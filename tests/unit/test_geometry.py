from caption_agent.domain.entities.bbox import BBox
from caption_agent.domain.entities.person_instance import PersonInstance
from caption_agent.infrastructure.imaging.geometry import normalized_distance, top_k_neighbors


def test_normalized_distance_non_negative() -> None:
    a = BBox(x1=10, y1=10, x2=30, y2=50)
    b = BBox(x1=40, y1=20, x2=60, y2=70)
    value = normalized_distance(a, b, image_w=100, image_h=100)
    assert value >= 0.0


def test_top_k_neighbors_excludes_target() -> None:
    target = PersonInstance(id="p1", bbox=BBox(x1=10, y1=10, x2=30, y2=60))
    p2 = PersonInstance(id="p2", bbox=BBox(x1=35, y1=10, x2=55, y2=60))
    p3 = PersonInstance(id="p3", bbox=BBox(x1=70, y1=10, x2=90, y2=60))
    neighbors = top_k_neighbors(target, [target, p2, p3], image_w=100, image_h=100, top_k=2)
    assert all(item.id != target.id for item in neighbors)
    assert len(neighbors) == 2

