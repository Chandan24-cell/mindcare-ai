from __future__ import annotations

from typing import Any, Dict, List, Optional


def estimate_cognitive_load(*, trend_window: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trend_window:
        return {
            "cognitive_load": "normal",
            "load_score": 0.3,
            "mental_fatigue_risk": "low",
        }

    stresses = [str(s.get("stress_level") or "").lower() for s in trend_window]
    high_ratio = sum(1 for s in stresses if s in {"high", "critical"}) / max(1, len(stresses))

    emotions = [str(s.get("emotion") or "").lower() for s in trend_window if s.get("emotion")]
    volatility = 0.0
    if len(emotions) >= 2:
        changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1])
        volatility = changes / max(1, len(emotions) - 1)

    wellness = [s.get("wellness_score") for s in trend_window if s.get("wellness_score") is not None]
    decline = 0.0
    if len(wellness) >= 2:
        half = len(wellness) // 2
        first = sum(wellness[:half]) / max(1, half)
        second = sum(wellness[half:]) / max(1, len(wellness) - half)
        decline = max(0.0, (first - second) / 40.0)

    load = 0.35 + 0.35 * high_ratio + 0.2 * volatility + 0.15 * decline
    load = max(0.0, min(1.0, load))

    cognitive_load = "elevated" if load >= 0.6 else "moderate" if load >= 0.45 else "normal"
    mental_fatigue_risk = "moderate" if load >= 0.5 else "low"

    return {
        "cognitive_load": cognitive_load,
        "load_score": round(float(load), 2),
        "mental_fatigue_risk": mental_fatigue_risk,
    }

