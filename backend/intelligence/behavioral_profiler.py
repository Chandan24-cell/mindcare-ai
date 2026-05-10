from __future__ import annotations

from typing import Any, Dict, List, Optional


def _stress_numeric(level: Optional[str]) -> float:
    s = str(level or "").strip().lower()
    return {"low": 0.15, "medium": 0.5, "high": 0.82, "critical": 0.95}.get(
        s, 0.5
    )


def _risk_profile_from_metrics(emotional_volatility: float, risk: float) -> str:
    v = max(0.0, min(1.0, float(risk)))
    if v < 0.33 and emotional_volatility < 0.45:
        return "low"
    if v < 0.6:
        return "moderate"
    if v < 0.8:
        return "high"
    return "critical"


def compute_behavioral_profile(*, trend_window: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute longitudinal behavioral intelligence profile."""
    if not trend_window:
        return {
            "behavioral_type": "stress_sensitive",
            "emotional_volatility": 0.5,
            "stress_resilience": 0.5,
            "wellness_consistency": 0.5,
            "risk_profile": "moderate",
            "adaptation_score": 0.5,
        }

    emotions = [str(s.get("emotion") or "").lower() for s in trend_window if s.get("emotion")]
    stress_levels = [s.get("stress_level") for s in trend_window if s.get("stress_level")]
    wellness_scores = [s.get("wellness_score") for s in trend_window if s.get("wellness_score") is not None]

    # volatility from unique emotion transitions
    if len(emotions) >= 2:
        changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1])
        emotional_volatility = min(1.0, changes / max(1, len(emotions) - 1))
    else:
        emotional_volatility = 0.5

    # wellness consistency from std dev mapped
    if len(wellness_scores) >= 3:
        import statistics

        std = float(statistics.pstdev([float(x) for x in wellness_scores]))
        wellness_consistency = max(0.0, 1.0 - std / 40.0)
    else:
        wellness_consistency = 0.6

    avg_stress = sum(_stress_numeric(x) for x in stress_levels) / max(1, len(stress_levels))
    stress_resilience = max(0.0, 1.0 - avg_stress)

    adaptation_score = max(0.0, min(1.0, 0.45 * stress_resilience + 0.55 * wellness_consistency))

    risk_profile = _risk_profile_from_metrics(emotional_volatility, avg_stress)

    behavioral_type = "stress_sensitive" if avg_stress > 0.55 else "resilient"

    return {
        "behavioral_type": behavioral_type,
        "emotional_volatility": round(float(emotional_volatility), 2),
        "stress_resilience": round(float(stress_resilience), 2),
        "wellness_consistency": round(float(wellness_consistency), 2),
        "risk_profile": risk_profile,
        "adaptation_score": round(float(adaptation_score), 2),
    }

