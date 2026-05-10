from __future__ import annotations

from typing import Any, Dict


def export_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Deterministic pass-through
    return payload

