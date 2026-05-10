from __future__ import annotations

from typing import Any, Dict, List, Optional


def _norm_emotion(x: Optional[str]) -> str:
    return str(x or "neutral").strip().lower()


def _trigger_from_history(stress_window: List[Dict[str, Any]]) -> str:
    # Heuristic: infer sleep_deprivation if sleep_hours trend low
    sleep_vals = [s.get("sleep_hours") for s in stress_window if s.get("sleep_hours") is not None]
    if len(sleep_vals) >= 3:
        if sum(sleep_vals[-3:]) / 3.0 < 5.0:
            return "sleep_deprivation"

    # otherwise: stress_level dominance
    stresses = [str(s.get("stress_level") or "").lower() for s in stress_window if s.get("stress_level")]
    if stresses:
        if any(x in {"high", "critical"} for x in stresses[-3:]):
            return "elevated_stress"
    return "general_stress"


def build_personalization(
    *,
    user_profile: Dict[str, Any],
    trend_window: List[Dict[str, Any]],
    modality_reliability: Dict[str, Any],
    burnout_risk: Optional[str],
) -> Dict[str, Any]:
    """Build personalization outputs deterministically."""

    baseline_emotion = _norm_emotion(user_profile.get("baseline_emotion"))
    preferred = user_profile.get("preferred_interventions") or ["breathing", "music"]

    trigger = _trigger_from_history(trend_window)

    # recovery speed mapping from recent recovery scores when available
    rec_scores = [
        s.get("recovery_score") for s in trend_window if s.get("recovery_score") is not None
    ]
    if rec_scores:
        avg = sum(rec_scores[-5:]) / max(1, len(rec_scores[-5:]))
        if avg >= 70:
            recovery_speed = "fast"
        elif avg >= 45:
            recovery_speed = "moderate"
        else:
            recovery_speed = "slow"
    else:
        recovery_speed = "moderate"

    # highest risk period: if timestamps exist, use hour-of-day histogram
    highest_risk_period = user_profile.get("highest_risk_period") or "unknown"
    timestamps = [s.get("timestamp") for s in trend_window if s.get("timestamp")]
    try:
        import datetime

        hours: List[int] = []
        for ts in timestamps[-10:]:
            try:
                dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                hours.append(dt.hour)
            except Exception:
                continue
        if hours:
            # late night: 0-5
            late = sum(1 for h in hours if 0 <= h <= 5)
            if late >= max(2, len(hours) // 3):
                highest_risk_period = "late_night"
            else:
                highest_risk_period = "evening"
    except Exception:
        pass

    behavioral_pattern = user_profile.get("behavioral_pattern") or "stress increases with low sleep"

    if burnout_risk in {"high", "critical"}:
        # prefer calming modalities when burnout elevated
        if "breathing" not in [p.lower() for p in preferred]:
            preferred = ["breathing"] + preferred

    # Modal reliability influence: emphasize dominant modality interventions.
    dominant_modality = str(modality_reliability.get("dominant_modality") or "").strip().lower()
    if dominant_modality == "sensor" and "music" not in [p.lower() for p in preferred]:
        preferred = preferred + ["music"]

    return {
        "baseline_emotion": baseline_emotion,
        "dominant_stress_trigger": trigger,
        "recovery_speed": recovery_speed,
        "preferred_interventions": preferred,
        "highest_risk_period": highest_risk_period,
        "behavioral_pattern": behavioral_pattern,
    }

