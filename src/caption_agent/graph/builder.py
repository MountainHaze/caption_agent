from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from caption_agent.domain.policies.verification_policy import VerificationPolicy
from caption_agent.graph.nodes.compose_summary import build_compose_summary_node
from caption_agent.graph.nodes.extract_attributes import build_extract_attributes_node
from caption_agent.graph.nodes.extract_object_relations import (
    build_extract_object_relations_node,
)
from caption_agent.graph.nodes.extract_person_relations import (
    build_extract_person_relations_node,
)
from caption_agent.graph.nodes.finalize_on_error import finalize_on_error_node
from caption_agent.graph.nodes.prepare_visual_context import prepare_visual_context_node
from caption_agent.graph.nodes.validate_input import validate_input_node
from caption_agent.graph.nodes.verify_facts import build_verify_facts_node
from caption_agent.graph.state import CaptionState
from caption_agent.infrastructure.llm.multimodal_client import MultimodalClientProtocol


def _route_after_validate(state: CaptionState) -> str:
    errors = state.get("errors", [])
    return "finalize_on_error" if errors else "prepare_visual_context"


def build_caption_graph(
    llm_client: MultimodalClientProtocol,
    policy: VerificationPolicy,
):
    graph_builder = StateGraph(CaptionState)
    graph_builder.add_node("validate_input", validate_input_node)
    graph_builder.add_node("finalize_on_error", finalize_on_error_node)
    graph_builder.add_node("prepare_visual_context", prepare_visual_context_node)
    graph_builder.add_node("extract_attributes", build_extract_attributes_node(llm_client))
    graph_builder.add_node(
        "extract_person_relations", build_extract_person_relations_node(llm_client)
    )
    graph_builder.add_node(
        "extract_object_relations", build_extract_object_relations_node(llm_client)
    )
    graph_builder.add_node("verify_facts", build_verify_facts_node(policy))
    graph_builder.add_node("compose_summary", build_compose_summary_node(llm_client))

    graph_builder.add_edge(START, "validate_input")
    graph_builder.add_conditional_edges(
        "validate_input",
        _route_after_validate,
        {
            "finalize_on_error": "finalize_on_error",
            "prepare_visual_context": "prepare_visual_context",
        },
    )
    graph_builder.add_edge("finalize_on_error", END)
    graph_builder.add_edge("prepare_visual_context", "extract_attributes")
    graph_builder.add_edge("extract_attributes", "extract_person_relations")
    graph_builder.add_edge("extract_person_relations", "extract_object_relations")
    graph_builder.add_edge("extract_object_relations", "verify_facts")
    graph_builder.add_edge("verify_facts", "compose_summary")
    graph_builder.add_edge("compose_summary", END)
    return graph_builder.compile()

