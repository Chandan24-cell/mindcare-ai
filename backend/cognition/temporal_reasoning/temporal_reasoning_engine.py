from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _stress_num(level: Optional[str]) -> float:
    s = str(level or "").lower().strip()
    return {"low": 0.15, "medium": 0.5, "high": 0.82, "critical": 0.95}.get(s, 0.5)


def compute_temporal_reasoning(*, window: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not window:
        return {
            "behavioral_momentum": "recovering",
            "trajectory_confidence": 0.5,
            "temporal_stability": 0.5,
            "wellness_direction": "improving",
        }

    # Wellness direction from trend in wellness_score if present, else from stress delta.
    wellness_scores = [
        s.get("wellness_score")
        for s in window
        if isinstance(s.get("wellness_score"), (int, float))
    ]

    if len(wellness_scores) >= 2:
        half = len(wellness_scores) // 2
        first = sum(wellness_scores[:half]) / max(1, half)
        second = sum(wellness_scores[half:]) / max(1, len(wellness_scores) - half)
        direction = "improving" if second - first >= 0 else "declining"
        stability = max(0.0, min(1.0, 1.0 - (abs(second - first) / 100.0)))
        momentum = "recovering" if direction == "improving" else "deteriorating"
        conf = max(0.0, min(1.0, 0.55 + 0.25 * stability))
        return {
            "behavioral_momentum": momentum,
            "trajectory_confidence": round(conf, 2),
            "temporal_stability": round(float(stability), 2),
            "wellness_direction": direction,
        }

    # Fallback: stress delta
    stresses = [_stress_num(s.get("stress_level")) for s in window]
    if len(stresses) >= 2:
        half = len(stresses) // 2
        first = sum(stresses[:half]) / max(1, half)
        second = sum(stresses[half:]) / max(1, len(stresses) - half)
        direction = "improving" if first - second >= 0 else "declining"
        stability = max(0.0, min(1.0, 1.0 - abs(first - second)))
        momentum = "recovering" if direction == "improving" else "deteriorating"
        conf = max(0.0, min(1.0, 0.5 + 0.35 * stability))
        return {
            "behavioral_momentum": momentum,
            "trajectory_confidence": round(conf, 2),
            "temporal_stability": round(float(stability), 2),
            "wellness_direction": direction,
        }

    return {
        "behavioral_momentum": "recovering",
        "trajectory_confidence": 0.81,
        "temporal_stability": 0.72,
        "wellness_direction": "improving",
    }

