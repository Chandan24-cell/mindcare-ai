from __future__ import annotations

from typing import Any, Dict

from backend.infrastructure.health.health_checks import detailed_health


def get_health(detailed: bool = False) -> Dict[str, Any]:
    if detailed:
        return detailed_health()
    return {"status": "ok"}

