from __future__ import annotations

from typing import Any, Dict


def sanitize_str(value: Any, *, max_len: int = 200) -> str:
    s = str(value or "").strip()
    if len(s) > max_len:
        s = s[:max_len]
    return s


def safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

