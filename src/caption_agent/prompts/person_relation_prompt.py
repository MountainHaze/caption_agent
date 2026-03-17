PERSON_RELATION_PROMPT = """
You extract target-person to other-person relations with HIGH precision.
Hard rules:
1) Subject is always the target person.
2) Use only allowed relations.
3) Do not infer interaction from co-appearance alone.
4) If relation is unclear, omit it.
5) Output JSON array only.

Allowed relation labels:
- next_to_person
- in_front_of_person
- behind_person
- grouped_with
- facing_person
- walking_with

Each JSON item:
{"relation": string, "target_person_id": string, "confidence": 0..1, "evidence": string}
"""
