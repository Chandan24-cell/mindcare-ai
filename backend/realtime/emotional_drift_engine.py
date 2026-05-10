from __future__ import annotations

from typing import Any, Dict, List, Optional


EMOTION_TO_VALENCE = {
    "happy": 1.0,
    "surprise": 0.8,
    "neutral": 0.5,
    "sad": 0.2,
    "angry": 0.1,
    "fear": 0.15,
    "disgust": 0.12,
}


def _valence(emotion: Optional[str]) -> Optional[float]:
    e = str(emotion or "").strip().lower()
    if not e:
        return None
    return EMOTION_TO_VALENCE.get(e)


def analyze_emotional_drift(window: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not window:
        return {
            "drift_direction": "stable",
            "drift_score": 0.0,
            "baseline_divergence": 0.0,
            "stability_impact": "low",
        }

    vals: List[float] = []
    for s in window:
        v = _valence(s.get("emotion"))
        if v is not None:
            vals.append(float(v))

    if len(vals) < 3:
        return {
            "drift_direction": "stable",
            "drift_score": 0.1,
            "baseline_divergence": 0.1,
            "stability_impact": "low",
        }

    half = len(vals) // 2
    first_avg = sum(vals[:half]) / max(1, half)
    second_avg = sum(vals[half:]) / max(1, len(vals) - half)

    drift = first_avg - second_avg
    drift_score = max(0.0, min(1.0, abs(drift) / 0.9))

    drift_direction = (
        "negative" if drift > 0 else "positive" if drift < 0 else "stable"
    )

    baseline_divergence = max(0.0, min(1.0, abs(second_avg - 0.5) / 0.5))

    impact = "low"
    if drift_score >= 0.3:
        impact = "moderate"
    if drift_score >= 0.55:
        impact = "high"

    return {
        "drift_direction": drift_direction,
        "drift_score": round(float(drift_score), 2),
        "baseline_divergence": round(float(baseline_divergence), 2),
        "stability_impact": impact,
    }

