from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.intelligence.modality_reliability import compute_modality_reliability


def _norm_emotion(x: Optional[str]) -> str:
    return str(x or "").strip().lower()


def _norm_stress(x: Optional[str]) -> str:
    return str(x or "").strip().lower()


def _factor_from_facial(facial: Dict[str, Any]) -> Optional[str]:
    emo = _norm_emotion(facial.get("emotion"))
    if emo in {"sad", "angry", "fear"}:
        return f"Facial emotion suggests {emo}".title()
    if emo in {"disgust"}:
        return "Facial emotion indicates aversion/discomfort"
    return f"Facial emotion is {emo or 'neutral'}".title()


def _factors_from_sensor(sensor: Dict[str, Any]) -> List[str]:
    bullets: List[str] = []
    if sensor.get("low_hrv"):
        bullets.append("Low HRV")
    if sensor.get("high_heart_rate"):
        bullets.append("Elevated heart rate")
    if sensor.get("low_sleep"):
        bullets.append("Reduced sleep duration")
    if sensor.get("low_activity_penalty"):
        bullets.append("Low activity/recovery fatigue")
    return bullets


def _secondary_factor(manual: Dict[str, Any]) -> Optional[str]:
    scale = manual.get("stress_scale")
    level = _norm_stress(manual.get("stress_level"))
    if scale is None and not level:
        return None
    if level:
        return f"Manual self-report indicates {level} stress".title()
    return f"Manual stress scale={scale}/10" if scale is not None else None


def build_explanation(
    *,
    facial_analysis: Dict[str, Any],
    sensor_analysis: Dict[str, Any],
    manual_analysis: Dict[str, Any],
    fusion_outputs: Dict[str, Any],
    trend_analytics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create structured deterministic explanation bullets."""

    facial_stress = facial_analysis.get("stress_level")
    sensor_stress = sensor_analysis.get("stress_level")
    manual_stress = manual_analysis.get("stress_level")

    reliability = compute_modality_reliability(
        facial_confidence=facial_analysis.get("confidence"),
        sensor_confidence=sensor_analysis.get("confidence"),
        manual_confidence=manual_analysis.get("confidence"),
        facial_stress_level=facial_stress,
        sensor_stress_level=sensor_stress,
        manual_stress_level=manual_stress,
    )

    facial_factor = _factor_from_facial(facial_analysis)
    sensor_factors = _factors_from_sensor(sensor_analysis)
    sec = _secondary_factor(manual_analysis)

    # Primary factor: choose highest impact based on flags and fusion severity
    primary_factor = None
    if sensor_analysis.get("low_hrv"):
        primary_factor = "Low HRV"
    elif sensor_analysis.get("low_sleep"):
        primary_factor = "Low sleep hours"
    elif sensor_analysis.get("high_heart_rate"):
        primary_factor = "Elevated heart rate"
    elif _norm_emotion(facial_analysis.get("emotion")) in {"sad", "angry", "fear", "disgust"}:
        primary_factor = "Negative facial emotion"

    if not primary_factor:
        primary_factor = "Fused multimodal stress indicators"

    secondary_factor = sec or facial_factor

    supporting_factors: List[str] = []
    # take up to 3 deterministic bullets
    for b in sensor_factors:
        if b not in supporting_factors:
            supporting_factors.append(b)
        if len(supporting_factors) >= 3:
            break

    # Add manual supporting if present
    if len(supporting_factors) < 3 and manual_analysis.get("stress_scale") is not None:
        supporting_factors.append(f"Manual stress scale={manual_analysis.get('stress_scale')}/10")

    if len(supporting_factors) < 3 and facial_analysis.get("emotion"):
        supporting_factors.append(f"Facial emotion={_norm_emotion(facial_analysis.get('emotion'))}")

    confidence_reason = "Multiple modalities aligned" if reliability.get("overall_modality_agreement", 0) >= 0.75 else "Modalities partially aligned"

    explanation_summary = (
        f"Stress level increased due to {primary_factor.lower()} and supporting indicators." 
        if str(fusion_outputs.get("stress_risk")) in {"high", "critical"}
        else f"Wellness state reflects {fusion_outputs.get('stress_risk','medium')} risk driven by {primary_factor.lower()}."
    )

    return {
        "primary_factor": primary_factor,
        "secondary_factor": secondary_factor,
        "supporting_factors": supporting_factors,
        "confidence_reason": confidence_reason,
        "modality_agreement": float(reliability.get("overall_modality_agreement", 0.6)),
        "explanation_summary": explanation_summary,
    }

