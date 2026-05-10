from __future__ import annotations

from typing import Any, Dict, List


def forecast_risk(
    *,
    trend_window: List[Dict[str, Any]],
    emotional_drift: Dict[str, Any],
    burnout_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    if not trend_window:
        return {
            "future_risk_level": "low",
            "forecast_confidence": 0.4,
            "predicted_direction": "stable",
            "estimated_recovery_sessions": 0,
        }

    wellness_scores = [
        s.get("wellness_score")
        for s in trend_window
        if s.get("wellness_score") is not None
    ]

    wellness_delta = 0
    if len(wellness_scores) >= 2:
        half = len(wellness_scores) // 2
        first = sum(wellness_scores[:half]) / max(1, half)
        second = sum(wellness_scores[half:]) / max(1, len(wellness_scores) - half)
        wellness_delta = second - first

    burnout_risk = str((burnout_analysis or {}).get("burnout_risk") or "low").lower()
    drift_score = float((emotional_drift or {}).get("drift_score") or 0.0)

    risk = 0.35
    if burnout_risk in {"high", "critical"}:
        risk += 0.35
    if drift_score >= 0.35:
        risk += 0.2
    if wellness_delta < -5:
        risk += 0.15

    risk = max(0.0, min(1.0, risk))

    future_risk_level = "medium"
    if risk < 0.4:
        future_risk_level = "low"
    elif risk >= 0.75:
        future_risk_level = "high"

    predicted_direction = (
        "improving" if wellness_delta > 5 else "worsening" if wellness_delta < -5 else "stable"
    )

    forecast_confidence = min(0.95, 0.55 + 0.05 * min(10, len(trend_window)))

    estimated_recovery_sessions = 4
    if predicted_direction == "worsening":
        estimated_recovery_sessions = 6 if risk >= 0.7 else 4
    elif predicted_direction == "improving":
        estimated_recovery_sessions = 2

    return {
        "future_risk_level": future_risk_level,
        "forecast_confidence": round(float(forecast_confidence), 2),
        "predicted_direction": predicted_direction,
        "estimated_recovery_sessions": int(estimated_recovery_sessions),
    }

