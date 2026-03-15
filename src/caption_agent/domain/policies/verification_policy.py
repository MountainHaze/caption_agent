from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from caption_agent.domain.entities.relation_result import RelationFact


@dataclass(frozen=True)
class VerificationPolicy:
    person_relation_threshold: float = 0.65
    object_relation_threshold: float = 0.75

    allowed_person_relations: tuple[str, ...] = (
        "next_to_person",
        "in_front_of_person",
        "behind_person",
        "grouped_with",
        "facing_person",
        "walking_with",
    )

    allowed_object_relations: tuple[str, ...] = (
        "holding_object",
        "carrying_object",
        "near_vehicle",
        "sitting_on_object",
        "using_object",
    )

    def verify_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        checked = dict(attributes)
        visibility = checked.get("visibility", {})
        face_visible = bool(visibility.get("face_visible", False))
        gender = checked.get("gender_presentation", {})
        if isinstance(gender, dict) and not face_visible:
            current = float(gender.get("confidence", 0.0))
            gender["confidence"] = min(current, 0.6)
            checked["gender_presentation"] = gender
        return checked

    def verify_person_relations(self, relations: list[dict[str, Any]]) -> list[RelationFact]:
        accepted: list[RelationFact] = []
        for rel in relations:
            label = rel.get("relation")
            confidence = float(rel.get("confidence", 0.0))
            if label not in self.allowed_person_relations:
                continue
            if confidence < self.person_relation_threshold:
                continue
            accepted.append(RelationFact(**rel))
        return accepted

    def verify_object_relations(self, relations: list[dict[str, Any]]) -> list[RelationFact]:
        accepted: list[RelationFact] = []
        for rel in relations:
            label = rel.get("relation")
            confidence = float(rel.get("confidence", 0.0))
            if label not in self.allowed_object_relations:
                continue
            if confidence < self.object_relation_threshold:
                continue
            accepted.append(RelationFact(**rel))
        return accepted

