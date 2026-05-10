from __future__ import annotations

from typing import Any, Dict, List


def simulate_future_state(*, window: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not window:
        return {
            "predicted_future_state": "stable_recovery",
            "burnout_probability": 0.21,
            "recovery_probability": 0.74,
            "future_confidence": 0.78,
        }

    wellness_scores = [s.get("wellness_score") for s in window if isinstance(s.get("wellness_score"), (int, float))]
    stress_levels = [str(s.get("stress_level") or "").lower() for s in window]

    burnout_prob = 0.3
    if wellness_scores:
        avg = sum(wellness_scores) / max(1, len(wellness_scores))
        burnout_prob = max(0.05, min(0.6, 0.45 - (avg / 200.0)))

    high_stress_ratio = sum(1 for s in stress_levels if s in {"high", "critical"}) / max(1, len(stress_levels))
    burnout_prob = max(0.0, min(0.9, burnout_prob + 0.2 * high_stress_ratio))

    recovery_prob = max(0.0, min(1.0, 0.9 - burnout_prob + 0.05))

    future_conf = max(0.4, min(0.92, 0.6 + 0.15 * (1.0 - high_stress_ratio)))

    predicted_state = "stable_recovery" if recovery_prob >= 0.65 else "unstable_decline"

    return {
        "predicted_future_state": predicted_state,
        "burnout_probability": round(float(burnout_prob), 2),
        "recovery_probability": round(float(recovery_prob), 2),
        "future_confidence": round(float(future_conf), 2),
    }

