from __future__ import annotations

from typing import Any, Dict, Optional


def analyze_micro_expression(
    *,
    stabilized_emotion: str,
    landmark_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Lightweight heuristic micro-expression analysis.

    Works even with single-frame input; deterministic fallback if metrics missing.
    """

    m = landmark_metrics or {}
    mouth_tension = m.get("mouth_tension")
    brow_stress = m.get("brow_stress")
    facial_symmetry = m.get("facial_symmetry")
    eye_openness = m.get("eye_openness")

    def _n(x: Any, default: float = 0.5) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return default

    mouth_tension = _n(mouth_tension, 0.45)
    brow_stress = _n(brow_stress, 0.4)
    facial_symmetry = _n(facial_symmetry, 0.7)
    eye_openness = _n(eye_openness, 0.55)

    score = 0.25
    score += 0.35 * brow_stress
    score += 0.25 * mouth_tension
    score += 0.15 * (1.0 - facial_symmetry)
    score += 0.10 * (1.0 - eye_openness)

    if stabilized_emotion in {"sad", "angry", "concerned", "overwhelmed"}:
        score = min(1.0, score + 0.08)

    hidden_stress = 0.2 + 0.55 * mouth_tension + 0.30 * brow_stress
    hidden_stress = max(0.0, min(1.0, hidden_stress))

    micro_detected = score >= 0.55

    return {
        "micro_expression_detected": bool(micro_detected),
        "micro_expression_score": round(score, 2),
        "hidden_stress_probability": round(hidden_stress, 2),
    }

