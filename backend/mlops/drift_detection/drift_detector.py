from __future__ import annotations

from typing import Any, Dict


def detect_drift(*, recent_predictions: Any) -> Dict[str, Any]:
    return {"drift_detected": False, "details": {}}

