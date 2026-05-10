from __future__ import annotations

from typing import Dict


def get_default_thresholds() -> Dict[str, float]:
    """Deterministic thresholds used by the affective pipeline."""
    return {
        "micro_expression_min_score": 0.55,
        "stability_variance_max": 0.28,
        "uncertainty_penalty": 0.35,
        "low_quality_reliability_cutoff": 0.45,
    }

