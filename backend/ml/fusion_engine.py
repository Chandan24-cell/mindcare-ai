from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from backend.ml.scoring_engine import (
    build_fusion_level_summary,
    score_confidence,
    score_wellness_and_stress,
    severity_from_wellness_score,
    stress_risk_label_from_score,
)
from backend.ml.reasoning_engine import generate_reasoning
from backend.ml.recommendation_engine import generate_recommendation_priority


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FacialAnalysis:
    emotion: Optional[str]
    stress_level: Optional[str]
    confidence: Optional[float]


@dataclass(frozen=True)
class SensorAnalysis:
    heart_rate: Optional[float]
    hrv: Optional[float]
    sleep_hours: Optional[float]
    activity_level: Optional[str]
    stress_level: Optional[str]
    confidence: Optional[float]


@dataclass(frozen=True)
class ManualAnalysis:
    mood: Optional[str]
    stress_scale: Optional[int]
    stress_level: Optional[str]
    confidence: Optional[float]


def _norm_emotion(x: Optional[str]) -> str:
    return str(x or "").strip().lower()


def _norm_stress_level(x: Optional[str]) -> str:
    return str(x or "").strip().lower()


def _stress_weight_from_facial_stress(stress_level: Optional[str]) -> float:
    s = _norm_stress_level(stress_level)
    mapping = {"low": 0.15, "medium": 0.45, "high": 0.75}
    return mapping.get(s, 0.45)


def _stress_weight_from_facial_emotion(emotion: Optional[str]) -> float:
    e = _norm_emotion(emotion)
    # Rule-based stress weighting per requirements.
    if e in {"sad", "angry", "fear"}:
        return 0.78
    if e in {"disgust"}:
        return 0.72
    if e in {"neutral"}:
        return 0.45
    if e in {"happy", "surprise"}:
        return 0.25
    return 0.45


def _sensor_signal_flags(
    *,
    heart_rate: Optional[float],
    hrv: Optional[float],
    sleep_hours: Optional[float],
    activity_level: Optional[str],
) -> Dict[str, bool]:
    """Return boolean flags based on existing inference heuristics.

    These flags are for reasoning + weighting and must be conservative.
    """
    hr = heart_rate
    hrv = hrv
    sleep = sleep_hours
    activity = _norm_emotion(activity_level)

    low_hrv = hrv is not None and hrv < 50
    low_sleep = sleep is not None and sleep < 5.0
    high_heart_rate = hr is not None and hr > 85
    low_activity_penalty = activity in {"low"}  # matches inference labels: low/moderate/high

    return {
        "low_hrv": bool(low_hrv),
        "low_sleep": bool(low_sleep),
        "high_heart_rate": bool(high_heart_rate),
        "low_activity_penalty": bool(low_activity_penalty),
    }


def _stress_weight_from_sensor_flags(flags: Dict[str, bool]) -> float:
    stress = 0.20
    # Each flag increases stress risk weighting
    if flags.get("low_hrv"):
        stress += 0.30
    if flags.get("high_heart_rate"):
        stress += 0.18
    # Sleep and activity are more wellness-penalty than pure stress
    if flags.get("low_sleep"):
        stress += 0.10
    if flags.get("low_activity_penalty"):
        stress += 0.10
    return min(stress, 1.0)


def _wellness_penalty_from_sensor_flags(flags: Dict[str, bool]) -> float:
    penalty = 0.10
    if flags.get("low_sleep"):
        penalty += 0.32
    if flags.get("low_activity_penalty"):
        penalty += 0.22
    # HRV is both stress and wellness regulation
    if flags.get("low_hrv"):
        penalty += 0.20
    if flags.get("high_heart_rate"):
        penalty += 0.12
    return min(penalty, 1.0)


def _stress_weight_from_manual(stress_scale: Optional[int], stress_level: Optional[str]) -> float:
    if stress_level:
        s = _norm_stress_level(stress_level)
        return {"low": 0.20, "medium": 0.50, "high": 0.82}.get(s, 0.50)

    if stress_scale is None:
        return 0.50
    # self-reported scale: <4 low, >7 high per existing inference
    if stress_scale < 4:
        return 0.20
    if stress_scale > 7:
        return 0.82
    return 0.50


def _wellness_penalty_from_manual(stress_scale: Optional[int], stress_level: Optional[str]) -> float:
    # Penalty mirrors stress; keep distinct but related
    return _stress_weight_from_manual(stress_scale, stress_level)


def _detect_conflict(
    *,
    facial_stress: Optional[str],
    sensor_stress: Optional[str],
    manual_stress: Optional[str],
) -> bool:
    """Detect opposing stress directions to adjust confidence."""

    vals = [_norm_stress_level(x) for x in (facial_stress, sensor_stress, manual_stress) if x]
    if len(vals) < 2:
        return False

    # Conflict if we see both low and high categories.
    if ("low" in vals and "high" in vals) or ("critical" in vals and "low" in vals):
        return True
    return False


def fuse_modalities(
    *,
    facial: Optional[Dict[str, Any]] = None,
    sensor: Optional[Dict[str, Any]] = None,
    manual: Optional[Dict[str, Any]] = None,
    history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Multimodal fusion entrypoint.

    - Inputs are dict-like and all optional.
    - Output is a structured dict with advanced fields.
    - Backward-compatibility: existing routes will continue to provide
      legacy fields (emotion/stress_level/confidence/reason/suggestion).
    """

    facial = facial or {}
    sensor = sensor or {}
    manual = manual or {}
    history = history or {}

    facial_available = bool(facial)
    sensor_available = bool(sensor)
    manual_available = bool(manual)

    facial_emotion = facial.get("emotion")
    facial_stress_level = facial.get("stress_level")
    facial_confidence = facial.get("confidence")

    sensor_stress_level = sensor.get("stress_level")
    sensor_confidence = sensor.get("confidence")

    manual_stress_scale = manual.get("stress_scale")
    manual_stress_level = manual.get("stress_level")
    manual_confidence = manual.get("confidence")

    # Sensor raw signals for flags (best-effort)
    sensor_hr = sensor.get("heart_rate")
    sensor_hrv = sensor.get("hrv")
    sensor_sleep = sensor.get("sleep_hours")
    sensor_activity = sensor.get("activity_level")

    flags = _sensor_signal_flags(
        heart_rate=sensor_hr,
        hrv=sensor_hrv,
        sleep_hours=sensor_sleep,
        activity_level=sensor_activity,
    )

    facial_stress_weight = _stress_weight_from_facial_stress(facial_stress_level)
    facial_emotion_weight = _stress_weight_from_facial_emotion(facial_emotion)
    facial_stress_signal = 0.60 * facial_stress_weight + 0.40 * facial_emotion_weight

    sensor_stress_signal = _stress_weight_from_sensor_flags(flags)
    sensor_wellness_penalty = _wellness_penalty_from_sensor_flags(flags)

    manual_stress_signal = _stress_weight_from_manual(
        manual_stress_scale, manual_stress_level
    )
    manual_wellness_penalty = _wellness_penalty_from_manual(
        manual_stress_scale, manual_stress_level
    )

    # Weighted fusion (configurable weights; can later be driven by settings)
    weights = {
        "facial": 0.33,
        "sensor": 0.42,
        "manual": 0.55,
    }

    # Normalize weights by availability
    available_weights = {
        k: v
        for k, v in weights.items()
        if (k == "facial" and facial_available)
        or (k == "sensor" and sensor_available)
        or (k == "manual" and manual_available)
    }
    wsum = sum(available_weights.values()) if available_weights else 1.0

    w_facial = available_weights.get("facial", 0.0) / wsum
    w_sensor = available_weights.get("sensor", 0.0) / wsum
    w_manual = available_weights.get("manual", 0.0) / wsum

    stress_risk_weighted = (
        w_facial * facial_stress_signal
        + w_sensor * sensor_stress_signal
        + w_manual * manual_stress_signal
    )

    wellness_penalty_weighted = (
        w_facial * 0.35 * facial_stress_signal  # facial contributes less to wellness penalty
        + w_sensor * sensor_wellness_penalty
        + w_manual * manual_wellness_penalty
    )

    conflicting_signals = _detect_conflict(
        facial_stress=facial_stress_level,
        sensor_stress=sensor_stress_level,
        manual_stress=manual_stress_level,
    )

    confidence = score_confidence(
        facial_available=facial_available,
        sensor_available=sensor_available,
        manual_available=manual_available,
        facial_confidence=float(facial_confidence) if facial_confidence is not None else None,
        sensor_confidence=float(sensor_confidence) if sensor_confidence is not None else None,
        manual_confidence=float(manual_confidence) if manual_confidence is not None else None,
    )

    # Reduce confidence slightly on conflicts
    if conflicting_signals:
        confidence = max(0.0, confidence - 0.12)

    scores = score_wellness_and_stress(
        stress_risk_weighted=stress_risk_weighted,
        wellness_penalty_weighted=wellness_penalty_weighted,
        confidence=confidence,
    )

    levels = build_fusion_level_summary(
        wellness_score=scores.wellness_score,
        stress_risk_score=scores.stress_risk_score,
    )

    severity = levels["severity"]
    stress_risk = levels["stress_risk"]

    # mental_state is intentionally high-level for UI/analytics
    # Map severity to mental state labels.
    mental_state = {
        "low": "calm",
        "moderate": "steady",
        "high": "concerned",
        "critical": "overwhelmed",
    }.get(severity, "steady")

    confidence = round(float(scores.confidence), 2)
    wellness_score = int(round(float(scores.wellness_score)))

    recommendation_priority = generate_recommendation_priority(
        severity=severity,
        stress_risk=stress_risk,
        confidence=confidence,
    )

    facial_analysis = {
        "emotion": facial_emotion,
        "stress_level": facial_stress_level,
        "confidence": facial_confidence,
        "stress_signal": round(float(facial_stress_signal), 3),
    }

    sensor_analysis = {
        "stress_level": sensor_stress_level,
        "confidence": sensor_confidence,
        "flags": flags,
        "stress_signal": round(float(sensor_stress_signal), 3),
        "wellness_penalty": round(float(sensor_wellness_penalty), 3),
    }

    manual_analysis = {
        "mood": manual.get("mood"),
        "stress_scale": manual_stress_scale,
        "stress_level": manual_stress_level,
        "confidence": manual_confidence,
        "stress_signal": round(float(manual_stress_signal), 3),
    }

    reasoning = generate_reasoning(
        mental_state=mental_state,
        stress_risk=stress_risk,
        wellness_score=wellness_score,
        facial_analysis=facial_analysis,
        sensor_analysis=sensor_analysis,
        manual_analysis=manual_analysis,
        conflicting_signals=conflicting_signals,
    )

    # history support placeholder (future extensibility)
    history_context = {
        "available": bool(history),
        "notes": "History fusion not yet applied; architecture reserved for future session/trend logic.",
    }

    return {
        "mental_state": mental_state,
        "stress_risk": stress_risk,
        "wellness_score": wellness_score,
        "confidence": confidence,
        "severity": severity,
        "reasoning": reasoning,
        "recommendation_priority": recommendation_priority,
        "modality_summary": {
            "facial_analysis": facial_analysis,
            "sensor_analysis": sensor_analysis,
            "manual_analysis": manual_analysis,
            "history": history_context,
        },
    }

