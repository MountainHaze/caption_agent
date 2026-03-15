from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from caption_agent.domain.entities.person_instance import PersonInstance
from caption_agent.infrastructure.imaging.geometry import iou, normalized_distance
from caption_agent.infrastructure.llm.schemas import AttributesPayload, RelationPayload
from caption_agent.infrastructure.observability.logging import get_logger

logger = get_logger("caption_agent.llm")


class MultimodalClientProtocol(Protocol):
    def extract_attributes(
        self,
        image_context: dict[str, Any],
        target: PersonInstance,
        language: str = "zh",
    ) -> dict[str, Any]:
        ...

    def extract_person_relations(
        self,
        image_context: dict[str, Any],
        target: PersonInstance,
        neighbors: list[PersonInstance],
    ) -> list[dict[str, Any]]:
        ...

    def extract_object_relations(
        self,
        image_context: dict[str, Any],
        target: PersonInstance,
    ) -> list[dict[str, Any]]:
        ...

    def compose_summary(
        self,
        attributes: dict[str, Any],
        person_relations: list[dict[str, Any]],
        object_relations: list[dict[str, Any]],
        language: str = "zh",
    ) -> str:
        ...


@dataclass
class LangChainMultimodalClient:
    """
    Framework-first client for MVP v2.

    This keeps logic in LangChain Runnable form and validates outputs with
    Pydantic schemas. You can replace the default runnables with model-backed
    chains without changing application or graph layers.
    """

    attribute_runnable: Runnable[dict[str, Any], dict[str, Any]] = field(
        default_factory=lambda: RunnableLambda(
            LangChainMultimodalClient._default_attributes_invoke
        )
    )
    person_relation_runnable: Runnable[dict[str, Any], list[dict[str, Any]]] = field(
        default_factory=lambda: RunnableLambda(
            LangChainMultimodalClient._default_person_relations_invoke
        )
    )
    object_relation_runnable: Runnable[dict[str, Any], list[dict[str, Any]]] = field(
        default_factory=lambda: RunnableLambda(
            LangChainMultimodalClient._default_object_relations_invoke
        )
    )
    summary_runnable: Runnable[dict[str, Any], str] = field(
        default_factory=lambda: RunnableLambda(
            LangChainMultimodalClient._default_summary_invoke
        )
    )

    def extract_attributes(
        self,
        image_context: dict[str, Any],
        target: PersonInstance,
        language: str = "zh",
    ) -> dict[str, Any]:
        payload = self.attribute_runnable.invoke(
            {"image_context": image_context, "target": target, "language": language}
        )
        try:
            return AttributesPayload.model_validate(payload).model_dump()
        except ValidationError:
            repaired_payload = self._repair_attributes_payload(payload)
            try:
                return AttributesPayload.model_validate(repaired_payload).model_dump()
            except ValidationError:
                logger.exception(
                    "Attributes payload validation failed after repair; using safe fallback. payload=%s repaired=%s",
                    payload,
                    repaired_payload,
                )
                return self._default_attributes_invoke(
                    {"image_context": image_context, "target": target}
                )

    def extract_person_relations(
        self,
        image_context: dict[str, Any],
        target: PersonInstance,
        neighbors: list[PersonInstance],
    ) -> list[dict[str, Any]]:
        payloads = self.person_relation_runnable.invoke(
            {
                "image_context": image_context,
                "target": target,
                "neighbors": neighbors,
            }
        )
        try:
            return [RelationPayload.model_validate(item).model_dump() for item in payloads]
        except ValidationError:
            logger.exception(
                "Person relations payload validation failed; using empty fallback. payload=%s",
                payloads,
            )
            return []

    def extract_object_relations(
        self,
        image_context: dict[str, Any],
        target: PersonInstance,
    ) -> list[dict[str, Any]]:
        payloads = self.object_relation_runnable.invoke(
            {"image_context": image_context, "target": target}
        )
        try:
            return [RelationPayload.model_validate(item).model_dump() for item in payloads]
        except ValidationError:
            logger.exception(
                "Object relations payload validation failed; using empty fallback. payload=%s",
                payloads,
            )
            return []

    def compose_summary(
        self,
        attributes: dict[str, Any],
        person_relations: list[dict[str, Any]],
        object_relations: list[dict[str, Any]],
        language: str = "zh",
    ) -> str:
        return self.summary_runnable.invoke(
            {
                "attributes": attributes,
                "person_relations": person_relations,
                "object_relations": object_relations,
                "language": language,
            }
        )

    @staticmethod
    def _default_attributes_invoke(payload: dict[str, Any]) -> dict[str, Any]:
        image_context = payload["image_context"]
        target: PersonInstance = payload["target"]

        image_w = int(image_context.get("image_width", 1))
        image_h = int(image_context.get("image_height", 1))
        area_ratio = target.bbox.area / max(float(image_w * image_h), 1.0)
        full_body = area_ratio > 0.08
        face_visible = area_ratio > 0.02

        return {
            "gender_presentation": {"value": "uncertain", "confidence": 0.35},
            "age_group": {"value": "uncertain", "confidence": 0.35},
            "visibility": {
                "full_body": full_body,
                "face_visible": face_visible,
                "occlusion_level": "low" if area_ratio > 0.05 else "medium",
            },
            "clothing": {
                "upper_garment": "unknown",
                "lower_garment": "unknown",
                "outerwear": None,
                "footwear": None,
                "accessories": [],
            },
            "appearance": {
                "hair": None,
                "pose": "standing" if full_body else "unknown",
                "orientation": "unknown",
            },
        }

    @staticmethod
    def _default_person_relations_invoke(payload: dict[str, Any]) -> list[dict[str, Any]]:
        image_context = payload["image_context"]
        target: PersonInstance = payload["target"]
        neighbors: list[PersonInstance] = payload["neighbors"]
        image_w = int(image_context.get("image_width", 1))
        image_h = int(image_context.get("image_height", 1))

        relations: list[dict[str, Any]] = []
        target_cx, _ = target.bbox.center
        for neighbor in neighbors:
            n_dist = normalized_distance(target.bbox, neighbor.bbox, image_w, image_h)
            overlap = iou(target.bbox, neighbor.bbox)

            if n_dist < 0.18:
                relations.append(
                    {
                        "relation": "next_to_person",
                        "target_person_id": neighbor.id,
                        "confidence": max(0.66, 0.9 - n_dist),
                        "evidence": "person centers are close in scene geometry",
                    }
                )
            if overlap > 0.02:
                relations.append(
                    {
                        "relation": "grouped_with",
                        "target_person_id": neighbor.id,
                        "confidence": min(0.85, 0.6 + overlap),
                        "evidence": "person boxes overlap or tightly cluster",
                    }
                )
            neighbor_cx, _ = neighbor.bbox.center
            if abs(target_cx - neighbor_cx) < (0.08 * image_w):
                relation = (
                    "in_front_of_person"
                    if target.bbox.y2 > neighbor.bbox.y2
                    else "behind_person"
                )
                relations.append(
                    {
                        "relation": relation,
                        "target_person_id": neighbor.id,
                        "confidence": 0.55,
                        "evidence": "approximate vertical ordering from box bottoms",
                    }
                )
        return relations

    @staticmethod
    def _default_object_relations_invoke(payload: dict[str, Any]) -> list[dict[str, Any]]:
        _ = payload
        return []

    @staticmethod
    def _default_summary_invoke(payload: dict[str, Any]) -> str:
        attributes = payload["attributes"]
        person_relations = payload["person_relations"]
        language = payload.get("language", "zh")

        age = attributes.get("age_group", {}).get("value", "uncertain")
        gender = attributes.get("gender_presentation", {}).get("value", "uncertain")
        clothing = attributes.get("clothing", {})
        upper = clothing.get("upper_garment") or "unknown"
        lower = clothing.get("lower_garment") or "unknown"

        if language != "zh":
            if person_relations:
                rel = person_relations[0]
                return (
                    f"Target person appears {age}/{gender}, wearing {upper} and {lower}, "
                    f"with relation {rel.get('relation', 'unknown')}."
                )
            return (
                f"Target person appears {age}/{gender}, wearing {upper} and {lower}, "
                "with no high-confidence person relation."
            )

        if person_relations:
            rel = person_relations[0]
            target_person = rel.get("target_person_id", "another person")
            relation = rel.get("relation", "unknown")
            return (
                f"\u76ee\u6807\u4eba\u7269\u5e74\u9f84\u63a8\u65ad\u4e3a {age}\uff0c"
                f"\u6027\u522b\u5448\u73b0\u4e3a {gender}\uff0c"
                f"\u670d\u9970\u4e3a {upper}/{lower}\uff0c"
                f"\u4e0e {target_person} \u5b58\u5728 {relation} \u5173\u7cfb\u3002"
            )
        return (
            f"\u76ee\u6807\u4eba\u7269\u5e74\u9f84\u63a8\u65ad\u4e3a {age}\uff0c"
            f"\u6027\u522b\u5448\u73b0\u4e3a {gender}\uff0c"
            f"\u670d\u9970\u4e3a {upper}/{lower}\uff0c"
            "\u672a\u68c0\u6d4b\u5230\u9ad8\u7f6e\u4fe1\u5ea6\u4eba\u7269\u5173\u7cfb\u3002"
        )

    @staticmethod
    def _repair_attributes_payload(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        repaired = dict(payload)

        repaired["gender_presentation"] = LangChainMultimodalClient._repair_confidence_field(
            repaired.get("gender_presentation")
        )
        repaired["age_group"] = LangChainMultimodalClient._repair_confidence_field(
            repaired.get("age_group")
        )
        repaired["visibility"] = LangChainMultimodalClient._repair_visibility_field(
            repaired.get("visibility")
        )
        repaired["clothing"] = LangChainMultimodalClient._repair_clothing_field(
            repaired.get("clothing")
        )
        repaired["appearance"] = LangChainMultimodalClient._repair_appearance_field(
            repaired.get("appearance")
        )
        return repaired

    @staticmethod
    def _repair_confidence_field(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            output = dict(value)
            output.setdefault("value", "uncertain")
            output["confidence"] = LangChainMultimodalClient._clamp_confidence(
                output.get("confidence", 0.5)
            )
            return output
        if isinstance(value, str):
            return {"value": value.strip() or "uncertain", "confidence": 0.6}
        return {"value": "uncertain", "confidence": 0.3}

    @staticmethod
    def _repair_visibility_field(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            output = dict(value)
            output["full_body"] = bool(output.get("full_body", False))
            output["face_visible"] = bool(output.get("face_visible", False))
            output["occlusion_level"] = str(output.get("occlusion_level", "medium"))
            return output
        if isinstance(value, str):
            lowered = value.lower()
            full_body = "full" in lowered or "fully visible" in lowered
            face_visible = "face" in lowered or "fully visible" in lowered
            occlusion = "low" if "fully" in lowered else "medium"
            return {
                "full_body": full_body,
                "face_visible": face_visible,
                "occlusion_level": occlusion,
            }
        return {"full_body": False, "face_visible": False, "occlusion_level": "medium"}

    @staticmethod
    def _repair_clothing_field(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            output = dict(value)
            output.setdefault("upper_garment", None)
            output.setdefault("lower_garment", None)
            output.setdefault("outerwear", None)
            output.setdefault("footwear", None)
            accessories = output.get("accessories", [])
            output["accessories"] = accessories if isinstance(accessories, list) else []
            return output
        if isinstance(value, str):
            return {
                "upper_garment": value.strip() or "unknown",
                "lower_garment": None,
                "outerwear": None,
                "footwear": None,
                "accessories": [],
            }
        return {
            "upper_garment": "unknown",
            "lower_garment": "unknown",
            "outerwear": None,
            "footwear": None,
            "accessories": [],
        }

    @staticmethod
    def _repair_appearance_field(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            output = dict(value)
            output.setdefault("hair", None)
            output.setdefault("pose", "unknown")
            output.setdefault("orientation", "unknown")
            return output
        if isinstance(value, str):
            return {"hair": value.strip() or None, "pose": "unknown", "orientation": "unknown"}
        return {"hair": None, "pose": "unknown", "orientation": "unknown"}

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            num = float(value)
        except Exception:  # noqa: BLE001
            return 0.5
        return max(0.0, min(1.0, num))
