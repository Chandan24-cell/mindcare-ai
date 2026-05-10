from __future__ import annotations

from typing import Any, Dict, Optional

from backend.ml.affective.confidence_calibrator import calibrate_confidence as _cal


def confidence_calibration(
    *,
    raw_confidence: float,
    uncertainty_score: float,
    reliability_score: float,
    modality_reliability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper."""

    return _cal(
        raw_confidence=raw_confidence,
        uncertainty_score=uncertainty_score,
        reliability_score=reliability_score,
        modality_reliability=modality_reliability,
    )

