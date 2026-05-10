from __future__ import annotations

from typing import Any, Dict, List


def build_alerts(
    *,
    escalation: Dict[str, Any],
    anomaly: Dict[str, Any],
    drift: Dict[str, Any],
    burnout_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    burnout_risk = str((burnout_analysis or {}).get("burnout_risk") or "low").lower()
    if burnout_risk in {"high", "critical"}:
        alerts.append(
            {
                "type": "burnout_warning",
                "priority": "high" if burnout_risk == "critical" else "medium",
                "message": "Persistent stress elevation detected.",
                "action": "Reduce workload and prioritize recovery.",
            }
        )

    if anomaly.get("anomaly_detected"):
        alerts.append(
            {
                "type": "anomaly_detected",
                "priority": anomaly.get("severity") or "medium",
                "message": f"Anomaly detected: {anomaly.get('anomaly_type') or 'anomaly'}.",
                "action": "Consider a short reset routine (breathing + hydration).",
            }
        )

    if drift.get("drift_direction") == "negative" and float(drift.get("drift_score") or 0.0) >= 0.3:
        alerts.append(
            {
                "type": "emotional_instability",
                "priority": "medium",
                "message": "Emotional drift indicates worsening stability.",
                "action": "Use grounding techniques and review sleep/recuperation habits.",
            }
        )

    if escalation.get("escalation_detected"):
        alerts.append(
            {
                "type": "escalation",
                "priority": escalation.get("escalation_level") or "high",
                "message": f"Escalation detected: {escalation.get('escalation_reason')}",
                "action": escalation.get("recommended_action") or "wellness_intervention",
            }
        )

    return alerts

