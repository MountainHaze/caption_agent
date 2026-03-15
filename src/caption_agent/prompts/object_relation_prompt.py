OBJECT_RELATION_PROMPT = """
You extract weak target-person to object relations conservatively.
Hard rules:
1) Keep precision high; prefer empty output over guessing.
2) Only use allowed relations.
3) If object type is unclear, use "unknown".
4) Output JSON array only.

Allowed relation labels:
- holding_object
- carrying_object
- near_vehicle
- sitting_on_object
- using_object

Each JSON item:
{"relation": string, "object_type": string|null, "confidence": 0..1, "evidence": string}
"""
