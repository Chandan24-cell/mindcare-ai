from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FusionScores:
    """Intermediate scoring outputs for fusion."""

    stress_risk_score: float  # 0..100
    wellness_score: float  # 0..100
    confidence: float  # 0..1


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _map_stress_level_from_risk_score(stress_risk_score: float) -> str:
    v = clamp(stress_risk_score, 0.0, 100.0)
    if v < 30:
        return "low"
    if v < 60:
        return "medium"
    if v < 85:
        return "high"
    return "critical"


def score_confidence(
    *,
    facial_available: bool,
    sensor_available: bool,
    manual_available: bool,
    facial_confidence: Optional[float] = None,
    sensor_confidence: Optional[float] = None,
    manual_confidence: Optional[float] = None,
    base: float = 0.35,
) -> float:
    """Compute overall confidence.

    - More modalities available => higher confidence.
    - Higher per-modality confidence => higher confidence.
    - Conflicting stress directions will be handled by fusion_engine.
    """

    availability = sum(
        1 for x in (facial_available, sensor_available, manual_available) if x
    )
    availability_bonus = 0.15 * availability

    provided = [
        facial_confidence if facial_available else None,
        sensor_confidence if sensor_available else None,
        manual_confidence if manual_available else None,
    ]
    provided_vals = [
        v for v in provided if v is not None and isinstance(v, (int, float))
    ]
    if provided_vals:
        avg = sum(provided_vals) / len(provided_vals)
        # avg is expected 0..1; clamp for safety
        avg = clamp(float(avg), 0.0, 1.0)
        confidence = base + availability_bonus + 0.45 * avg
    else:
        confidence = base + availability_bonus + 0.15

    return clamp(confidence, 0.0, 1.0)


def score_wellness_and_stress(
    *,
    stress_risk_weighted: float,
    wellness_penalty_weighted: float,
    confidence: float,
) -> FusionScores:
    """Produce final numeric scores from weighted aggregates.

    Inputs are already normalized in fusion_engine (0..1 scales) but kept
    flexible here for future extension.
    """

    # stress_risk_weighted: 0..1 where higher => more stress risk
    # wellness_penalty_weighted: 0..1 where higher => worse wellness
    stress_risk_score = clamp(float(stress_risk_weighted) * 100.0, 0.0, 100.0)

    # wellness_score is inverse penalty, but still bounded by confidence
    base_wellness = 100.0 * (1.0 - clamp(float(wellness_penalty_weighted), 0.0, 1.0))
    # Reduce wellness score slightly if confidence is low (uncertainty)
    confidence_adjustment = (1.0 - clamp(confidence, 0.0, 1.0)) * 12.0
    wellness_score = clamp(base_wellness - confidence_adjustment, 0.0, 100.0)

    return FusionScores(
        stress_risk_score=stress_risk_score,
        wellness_score=wellness_score,
        confidence=clamp(confidence, 0.0, 1.0),
    )


def severity_from_wellness_score(wellness_score: float) -> str:
    v = clamp(float(wellness_score), 0.0, 100.0)
    if v >= 80:
        return "low"  # excellent
    if v >= 60:
        return "moderate"  # stable
    if v >= 40:
        return "high"  # moderate concern
    return "critical"  # high concern


def stress_risk_label_from_score(stress_risk_score: float) -> str:
    return _map_stress_level_from_risk_score(stress_risk_score)


def build_fusion_level_summary(
    *,
    wellness_score: float,
    stress_risk_score: float,
) -> Dict[str, Any]:
    return {
        "severity": severity_from_wellness_score(wellness_score),
        "stress_risk": stress_risk_label_from_score(stress_risk_score),
    }

