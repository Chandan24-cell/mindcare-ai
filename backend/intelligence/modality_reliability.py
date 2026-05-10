from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _norm_stress_level(x: Optional[str]) -> str:
    return str(x or "").strip().lower()


def _agreement_score(facial: Optional[str], sensor: Optional[str], manual: Optional[str]) -> float:
    vals = [_norm_stress_level(v) for v in (facial, sensor, manual) if v]
    if len(vals) < 2:
        return 0.6
    # If majority agrees, higher agreement.
    from collections import Counter

    c = Counter(vals)
    common = c.most_common(1)[0][1]
    return min(0.95, max(0.4, common / len(vals)))


def compute_modality_reliability(
    *,
    facial_confidence: Optional[float],
    sensor_confidence: Optional[float],
    manual_confidence: Optional[float],
    facial_stress_level: Optional[str],
    sensor_stress_level: Optional[str],
    manual_stress_level: Optional[str],
) -> Dict[str, Any]:
    """Compute deterministic modality reliability.

    Reliability is based on available confidences and agreement.
    """

    # If a modality is absent, keep its reliability low.
    def _rel(conf: Optional[float]) -> float:
        if conf is None:
            return 0.2
        try:
            c = float(conf)
        except Exception:
            return 0.2
        return min(1.0, max(0.15, c))

    facial_reliability = _rel(facial_confidence)
    sensor_reliability = _rel(sensor_confidence)
    manual_input_reliability = _rel(manual_confidence)

    overall_modality_agreement = _agreement_score(
        facial_stress_level,
        sensor_stress_level,
        manual_stress_level,
    )

    # dominant modality: highest reliability among available
    reliabilities = {
        "facial": facial_reliability,
        "sensor": sensor_reliability,
        "manual": manual_input_reliability,
    }
    dominant_modality = max(reliabilities.items(), key=lambda kv: kv[1])[0]

    return {
        "facial_reliability": round(facial_reliability, 2),
        "sensor_reliability": round(sensor_reliability, 2),
        "manual_input_reliability": round(manual_input_reliability, 2),
        "overall_modality_agreement": round(overall_modality_agreement, 2),
        "dominant_modality": dominant_modality,
    }

