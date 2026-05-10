from __future__ import annotations

from typing import Any, Dict

from backend.observability.metrics.metrics_registry import registry


def get_metrics_payload() -> Dict[str, Any]:
    """Return current in-process metrics snapshot."""
    return {"counters": registry.snapshot()}

