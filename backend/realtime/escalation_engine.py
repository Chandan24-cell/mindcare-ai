from __future__ import annotations

from typing import Any, Dict, List


def detect_escalation(
    *,
    trend_window: List[Dict[str, Any]],
    burnout_analysis: Dict[str, Any],
    drift_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    if not trend_window:
        return {
            "escalation_detected": False,
            "escalation_level": "low",
            "escalation_reason": "insufficient history",
            "recommended_action": "monitor",
        }

    burnout_risk = str((burnout_analysis or {}).get("burnout_risk") or "low").lower()

    stresses = [str(s.get("stress_level") or "").lower() for s in trend_window]
    high_ratio = sum(
        1 for s in stresses if s in {"high", "critical"}
    ) / max(1, len(stresses))

    drift_score = float((drift_analysis or {}).get("drift_score") or 0.0)
    drift_direction = str((drift_analysis or {}).get("drift_direction") or "").lower()

    escalation_score = 0.0
    if burnout_risk in {"high", "critical"}:
        escalation_score += 0.45
    if high_ratio >= 0.5:
        escalation_score += 0.35
    if drift_score >= 0.35 and drift_direction == "negative":
        escalation_score += 0.25

    escalation_detected = escalation_score >= 0.55

    escalation_level = "low"
    if escalation_score >= 0.65:
        escalation_level = "medium"
    if escalation_score >= 0.8:
        escalation_level = "high"

    if burnout_risk in {"high", "critical"}:
        reason = "persistent burnout indicators"
    elif high_ratio >= 0.5:
        reason = "repeated high stress patterns"
    else:
        reason = "emotional drift and stress escalation"

    return {
        "escalation_detected": bool(escalation_detected),
        "escalation_level": escalation_level,
        "escalation_reason": reason,
        "recommended_action": "wellness_intervention" if escalation_detected else "continue_monitoring",
    }

