from __future__ import annotations

from typing import Any

from caption_agent.graph.state import CaptionState
from caption_agent.infrastructure.llm.multimodal_client import MultimodalClientProtocol


def build_compose_summary_node(llm_client: MultimodalClientProtocol):
    def _node(state: CaptionState) -> dict[str, Any]:
        include_summary = bool(state.get("include_summary", True))
        summary = None
        if include_summary:
            summary = llm_client.compose_summary(
                attributes=state.get("verified_attributes", {}),
                person_relations=state.get("verified_person_relations", []),
                object_relations=state.get("verified_object_relations", []),
                language=state.get("language", "zh"),
            )

        result = {
            "instance_id": state.get("target_instance_id"),
            "attributes": state.get("verified_attributes", {}),
            "person_relations": state.get("verified_person_relations", []),
            "object_relations": state.get("verified_object_relations", []),
            "summary": summary,
        }
        return {"summary": summary, "final_result": result}

    return _node

