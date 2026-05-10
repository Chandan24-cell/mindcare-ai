from __future__ import annotations

from typing import Any, Dict, List, Optional


def compute_cognitive_state(*, window: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not window:
        return {
            "cognitive_resilience": 0.68,
            "decision_fatigue": "moderate",
            "mental_capacity": "stable",
        }

    stresses = [str(x.get("stress_level") or "").lower() for x in window]
    high_ratio = sum(1 for s in stresses if s in {"high", "critical"}) / max(1, len(stresses))

    # Emotional volatility proxy
    emotions = [str(x.get("emotion") or "").lower() for x in window if x.get("emotion")]
    volatility = 0.0
    if len(emotions) >= 2:
        changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1])
        volatility = changes / max(1, len(emotions) - 1)

    resilience = max(0.0, min(1.0, 0.82 - 0.35 * high_ratio - 0.2 * volatility))

    fatigue = "low"
    if high_ratio >= 0.45 or volatility >= 0.35:
        fatigue = "moderate"
    if high_ratio >= 0.7:
        fatigue = "high"

    mental_capacity = "stable" if resilience >= 0.6 else "limited"

    return {
        "cognitive_resilience": round(float(resilience), 2),
        "decision_fatigue": fatigue,
        "mental_capacity": mental_capacity,
    }

