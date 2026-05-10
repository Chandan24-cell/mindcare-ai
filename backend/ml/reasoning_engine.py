from __future__ import annotations

from typing import Any, Dict, List, Optional


def _norm_str(x: Optional[str]) -> str:
    return str(x or "").strip().lower()


def generate_reasoning(
    *,
    mental_state: str,
    stress_risk: str,
    wellness_score: float,
    facial_analysis: Dict[str, Any],
    sensor_analysis: Dict[str, Any],
    manual_analysis: Dict[str, Any],
    conflicting_signals: bool,
) -> List[str]:
    """Create contextual reasoning bullets.

    Keep it deterministic and non-LLM to avoid variability.
    """

    bullets: List[str] = []

    # Stress driver explanations
    if facial_analysis.get("stress_signal", 0.0) > 0.55:
        emotion = facial_analysis.get("emotion")
        bullets.append(f"Facial emotion ({emotion}) suggests elevated stress indicators.")

    if sensor_analysis.get("low_hrv", False):
        bullets.append("Low HRV contributed to higher stress-risk weighting.")

    if sensor_analysis.get("low_sleep", False):
        bullets.append("Low sleep hours contributed to reduced wellness score.")

    if sensor_analysis.get("high_heart_rate", False):
        bullets.append("Elevated heart rate increased anxiety/stress risk considerations.")

    if sensor_analysis.get("low_activity_penalty", False):
        bullets.append("Low activity level applied a wellness penalty due to recovery fatigue risk.")

    if manual_analysis.get("stress_scale", None) is not None:
        scale = manual_analysis.get("stress_scale")
        reported = manual_analysis.get("stress_level")
        bullets.append(
            f"Manual self-report (stress scale={scale}/10, interpreted as {reported}) strongly influenced the final fusion."
        )

    # Confidence / conflict
    if conflicting_signals:
        bullets.append("Conflicting modalities were detected; confidence was reduced and severity narrowed.")

    # Wellness score bracket
    if wellness_score >= 80:
        bullets.append("Overall wellness is excellent based on the fused indicators.")
    elif wellness_score >= 60:
        bullets.append("Overall wellness appears stable with moderate protective factors.")
    elif wellness_score >= 40:
        bullets.append("Overall wellness shows moderate concern; consider stress-reduction routines.")
    else:
        bullets.append("Overall wellness is high concern; prioritize recovery, breathing, and calming activities.")

    # Ensure at least one bullet
    if not bullets:
        bullets.append(f"Fusion result indicates {mental_state} with {stress_risk} stress risk.")

    return bullets

