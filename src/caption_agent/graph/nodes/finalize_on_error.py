from __future__ import annotations

from caption_agent.graph.state import CaptionState


def finalize_on_error_node(state: CaptionState) -> dict:
    return {
        "final_result": {
            "instance_id": state.get("target_instance_id", "unknown"),
            "attributes": {},
            "person_relations": [],
            "object_relations": [],
            "summary": None,
        }
    }

