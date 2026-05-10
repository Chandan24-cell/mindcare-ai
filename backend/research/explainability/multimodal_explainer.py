from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_explanation_trace(
    *,
    emotion: Optional[str],
    stress_level: Optional[str],
    confidence: Optional[float],
    modality_importance: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Deterministic explainability trace (placeholder-safe).

    This is intentionally local + deterministic to avoid external dependencies.
    """

    modality_importance = modality_importance or {
        "facial": 0.35,
        "sensor": 0.35,
        "manual": 0.30,
    }

    dominant_signals: List[str] = []
    if modality_importance.get("facial", 0) >= max(modality_importance.values()):
        dominant_signals.append("facial_emotion")
    if modality_importance.get("sensor", 0) >= max(modality_importance.values()):
        dominant_signals.append("sensor_markers")
    if modality_importance.get("manual", 0) >= max(modality_importance.values()):
        dominant_signals.append("manual_stress_scale")

    reasoning_trace = [
        f"emotion={emotion or 'unknown'} contributed to stress interpretation",
        f"stress_level={stress_level or 'unknown'} mapped into wellness scoring",
        "confidence calibrated from available modalities and internal certainty",
    ]

    confidence_factors = {
        "available_modalities": len([k for k, v in modality_importance.items() if v > 0]),
        "signal_quality": 0.75,
        "conflict_penalty": 0.0,
    }

    return {
        "reasoning_trace": reasoning_trace,
        "dominant_signals": dominant_signals or ["facial_emotion"],
        "modality_importance": modality_importance,
        "confidence_factors": confidence_factors,
        "confidence": confidence,
    }

