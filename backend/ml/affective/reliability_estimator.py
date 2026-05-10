from __future__ import annotations

from typing import Any, Dict, List

from backend.ml.affective.reliability_estimator import estimate_reliability as _er


def compute_prediction_reliability(
    *,
    raw_confidence: float,
    face_landmark_confidence: float | None,
    temporal_stability_score: float | None,
    quality_flags: List[str] | None = None,
) -> Dict[str, Any]:
    return _er(
        raw_confidence=raw_confidence,
        face_landmark_confidence=face_landmark_confidence,
        temporal_stability_score=temporal_stability_score,
        quality_flags=quality_flags,
    )

