from __future__ import annotations

import json
from typing import Any


def safe_json_loads(payload: str, fallback: Any) -> Any:
    try:
        return json.loads(payload)
    except Exception:
        return fallback

