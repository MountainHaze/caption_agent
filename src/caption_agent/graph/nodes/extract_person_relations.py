from __future__ import annotations

from typing import Any

from caption_agent.graph.state import CaptionState
from caption_agent.infrastructure.llm.multimodal_client import MultimodalClientProtocol


def build_extract_person_relations_node(llm_client: MultimodalClientProtocol):
    def _node(state: CaptionState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        try:
            relations = llm_client.extract_person_relations(
                image_context=state.get("image_context", {}),
                target=state["target_instance"],
                neighbors=state.get("neighbor_persons", []),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"extract_person_relations_failed: {exc}")
            relations = []
        return {"raw_person_relations": relations, "errors": errors}

    return _node

