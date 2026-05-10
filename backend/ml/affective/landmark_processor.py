from __future__ import annotations

from typing import Any, Dict

from backend.ml.affective.affective_models import FacialLandmarkMetrics


def process_landmark_metrics(metrics: FacialLandmarkMetrics) -> Dict[str, Any]:
    """Convert landmark metrics into facial-signal oriented dict."""
    d = metrics.to_dict()

    # Provide normalized derived signals.
    # mouth_tension => emotional tension proxy
    eye_openness = d.get("eye_openness")
    brow_stress = d.get("brow_stress")

    # Deterministic mappings
    fatigue_signal = 0.0
    if isinstance(eye_openness, (int, float)):
        fatigue_signal = 1.0 - float(eye_openness)

    engagement_signal = 0.0
    if isinstance(eye_openness, (int, float)):
        engagement_signal = float(eye_openness)

    cognitive_strain = 0.0
    if isinstance(brow_stress, (int, float)):
        cognitive_strain = float(brow_stress)

    d.update(
        {
            "fatigue_signal": fatigue_signal,
            "engagement_signal": engagement_signal,
            "cognitive_strain": cognitive_strain,
        }
    )

    return d

