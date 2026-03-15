from __future__ import annotations

from caption_agent.graph.state import CaptionState


def validate_input_node(state: CaptionState) -> dict:
    errors = list(state.get("errors", []))
    instances = state.get("instances", [])
    target_id = state.get("target_instance_id")
    if not instances:
        errors.append("instances cannot be empty")
        return {"errors": errors}
    if not target_id:
        errors.append("target_instance_id is required")
        return {"errors": errors}

    target = next((item for item in instances if item.id == target_id), None)
    if target is None:
        errors.append(f"target_instance_id={target_id} not found in instances")
        return {"errors": errors}

    return {"target_instance": target, "errors": errors}

