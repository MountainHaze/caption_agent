from __future__ import annotations

from typing import Any

from caption_agent.graph.state import CaptionState
from caption_agent.infrastructure.llm.multimodal_client import MultimodalClientProtocol


def build_extract_attributes_node(llm_client: MultimodalClientProtocol):
    def _node(state: CaptionState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        try:
            attrs = llm_client.extract_attributes(
                image_context=state.get("image_context", {}),
                target=state["target_instance"],
                language=state.get("language", "zh"),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"extract_attributes_failed: {exc}")
            attrs = {}
        return {"raw_attributes": attrs, "errors": errors}

    return _node

