from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _stress_weight(stress_level: Optional[str]) -> float:
    s = str(stress_level or "").lower()
    return {"low": 0.15, "medium": 0.5, "high": 0.82, "critical": 0.95}.get(
        s, 0.5
    )


def _risk_from_signals(signals: Dict[str, Any]) -> float:
    # Weighted, deterministic scoring.
    score = 0.0
    score += 0.35 * float(signals.get("persistent_high_stress", 0.0))
    score += 0.20 * float(signals.get("poor_sleep_trend", 0.0))
    score += 0.15 * float(signals.get("elevated_hr_trend", 0.0))
    score += 0.15 * float(signals.get("negativity_persistence", 0.0))
    score += 0.15 * float(signals.get("declining_wellness", 0.0))
    return max(0.0, min(1.0, score))


def burnout_level_from_score(score01: float) -> str:
    if score01 < 0.25:
        return "low"
    if score01 < 0.45:
        return "medium"
    if score01 < 0.7:
        return "high"
    return "critical"


NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}


def detect_burnout(signals_window: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], int]:
    """Detect burnout risk from last N sessions.

    Missing modality fields are tolerated.
    """
    if not signals_window:
        return "low", {"reason": "no history"}, 0

    stresses = [s.get("stress_level") for s in signals_window]
    wellness_scores = [s.get("wellness_score") for s in signals_window if s.get("wellness_score") is not None]

    high_count = sum(1 for st in stresses if str(st or "").lower() in {"high", "critical"})
    persistent_high_stress = high_count / max(1, len(stresses))

    # Sleep trend: compare first half vs second half means when available.
    sleep_vals = [s.get("sleep_hours") for s in signals_window if s.get("sleep_hours") is not None]
    poor_sleep_trend = 0.0
    if len(sleep_vals) >= 3:
        mid = len(sleep_vals) // 2
        first_avg = sum(sleep_vals[:mid]) / max(1, mid)
        second_avg = sum(sleep_vals[mid:]) / max(1, len(sleep_vals) - mid)
        poor_sleep_trend = 1.0 if second_avg < first_avg else 0.2

    # HR trend
    hr_vals = [s.get("heart_rate") for s in signals_window if s.get("heart_rate") is not None]
    elevated_hr_trend = 0.0
    if len(hr_vals) >= 3:
        mid = len(hr_vals) // 2
        first_avg = sum(hr_vals[:mid]) / max(1, mid)
        second_avg = sum(hr_vals[mid:]) / max(1, len(hr_vals) - mid)
        elevated_hr_trend = 1.0 if second_avg > first_avg else 0.2

    # Negativity persistence (fraction of negative emotions)
    emotions = [s.get("emotion") for s in signals_window]
    neg_count = sum(1 for e in emotions if str(e or "").lower() in NEGATIVE_EMOTIONS)
    negativity_persistence = neg_count / max(1, len(emotions))

    # Declining wellness
    declining_wellness = 0.0
    if len(wellness_scores) >= 3:
        half = len(wellness_scores) // 2
        first_avg = sum(wellness_scores[:half]) / max(1, half)
        second_avg = sum(wellness_scores[half:]) / max(1, len(wellness_scores) - half)
        declining_wellness = 1.0 if second_avg < first_avg else 0.2

    signals = {
        "persistent_high_stress": persistent_high_stress,
        "poor_sleep_trend": poor_sleep_trend,
        "elevated_hr_trend": elevated_hr_trend,
        "negativity_persistence": negativity_persistence,
        "declining_wellness": declining_wellness,
    }

    score01 = _risk_from_signals(signals)
    level = burnout_level_from_score(score01)
    score = int(round(score01 * 100))

    return level, signals, score

