SUMMARY_PROMPT = """
Compose a short factual summary from verified structured fields only.
Hard rules:
1) Do not add facts not present in input.
2) Keep it concise (1-2 sentences).
3) Match output language requested by caller.
4) If relation list is empty, say no high-confidence relation.
"""
