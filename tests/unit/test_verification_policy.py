from caption_agent.domain.policies.verification_policy import VerificationPolicy


def test_verify_person_relations_threshold() -> None:
    policy = VerificationPolicy(person_relation_threshold=0.7, object_relation_threshold=0.8)
    raw = [
        {
            "relation": "next_to_person",
            "target_person_id": "p2",
            "confidence": 0.71,
            "evidence": "close",
        },
        {
            "relation": "walking_with",
            "target_person_id": "p3",
            "confidence": 0.55,
            "evidence": "low",
        },
    ]
    checked = policy.verify_person_relations(raw)
    assert len(checked) == 1
    assert checked[0].relation == "next_to_person"


def test_verify_attributes_reduces_gender_confidence_when_face_missing() -> None:
    policy = VerificationPolicy()
    raw = {
        "gender_presentation": {"value": "male", "confidence": 0.92},
        "visibility": {"face_visible": False},
    }
    checked = policy.verify_attributes(raw)
    assert checked["gender_presentation"]["confidence"] <= 0.6

