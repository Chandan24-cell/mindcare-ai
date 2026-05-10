from __future__ import annotations

from typing import Any, Dict, List, Optional


def recommendation_priority_from_severity(severity: str) -> str:
    s = str(severity or "").lower().strip()
    if s in {"critical", "high"}:
        return "high"
    if s == "moderate":
        return "medium"
    return "low"


def generate_recommendation_priority(
    *,
    severity: str,
    stress_risk: str,
    confidence: float,
) -> str:
    """Map fused stress/severity to a priority label.

    Keep simple + deterministic.
    """
    base = recommendation_priority_from_severity(severity)
    if confidence < 0.5 and base != "high":
        return "medium"  # uncertainty => don't over-escalate
    return base


def build_recommendation_context(
    *,
    severity: str,
    recommendation_priority: str,
    modality_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "severity": severity,
        "recommendation_priority": recommendation_priority,
        "modality_summary": modality_summary,
    }

