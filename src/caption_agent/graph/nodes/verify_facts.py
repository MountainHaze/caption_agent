from __future__ import annotations

from typing import Any

from caption_agent.domain.policies.verification_policy import VerificationPolicy
from caption_agent.graph.state import CaptionState


def build_verify_facts_node(policy: VerificationPolicy):
    def _node(state: CaptionState) -> dict[str, Any]:
        attrs = policy.verify_attributes(state.get("raw_attributes", {}))
        person_relations = policy.verify_person_relations(
            state.get("raw_person_relations", [])
        )
        object_relations = policy.verify_object_relations(
            state.get("raw_object_relations", [])
        )
        return {
            "verified_attributes": attrs,
            "verified_person_relations": [item.model_dump() for item in person_relations],
            "verified_object_relations": [item.model_dump() for item in object_relations],
        }

    return _node

