from __future__ import annotations

from typing import Any, Dict

from backend.ml.affective.affective_models import FacialLandmarkMetrics


def extract_facial_signals(*, landmark_metrics: FacialLandmarkMetrics) -> Dict[str, Any]:
    """Extract interpretable facial signals.

    This is a deterministic mapping from landmark metrics to signals.
    """

    # Clamp helpers
    def _c(v: Any, default: float = 0.5) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return default

    eye_openness = _c(landmark_metrics.eye_openness, default=0.55)
    mouth_tension = _c(landmark_metrics.mouth_tension, default=0.45)
    brow_stress = _c(landmark_metrics.brow_stress, default=0.4)

    # Deterministic proxies
    fatigue_signal = round(1.0 - eye_openness, 2)
    engagement_signal = round(eye_openness, 2)
    cognitive_strain = round(0.55 * brow_stress + 0.45 * mouth_tension, 2)

    return {
        "fatigue_signal": fatigue_signal,
        "engagement_signal": engagement_signal,
        "cognitive_strain": cognitive_strain,
    }

