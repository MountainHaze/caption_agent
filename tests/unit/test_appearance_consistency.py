from caption_agent.infrastructure.llm.multimodal_client import LangChainMultimodalClient


def test_riding_activity_keeps_activity_field() -> None:
    payload = {
        "hair": "short dark hair",
        "activity": "riding a horse",
        "orientation": "front-facing",
    }
    repaired = LangChainMultimodalClient._repair_appearance_field(payload)
    assert repaired["activity"] == "riding a horse"
    assert "pose" not in repaired
