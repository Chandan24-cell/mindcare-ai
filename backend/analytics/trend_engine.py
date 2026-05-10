from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.analytics.burnout_detector import detect_burnout
from backend.analytics.recovery_engine import compute_recovery
from backend.analytics.stability_engine import compute_stability


def _stress_direction(stress_levels: List[str]) -> str:
    vals = [str(s or "").lower() for s in stress_levels]
    # Map to numeric
    numeric = [{"low": 0, "medium": 1, "high": 2, "critical": 3}.get(v, 1) for v in vals]
    if len(numeric) < 2:
        return "stable"

    half = len(numeric) // 2
    first_avg = sum(numeric[:half]) / max(1, half)
    second_avg = sum(numeric[half:]) / max(1, len(numeric) - half)

    if second_avg < first_avg:
        return "decreasing"
    if second_avg > first_avg:
        return "increasing"
    return "stable"


def _trend_direction(wellness_delta: int) -> str:
    if wellness_delta > 5:
        return "improving"
    if wellness_delta < -5:
        return "worsening"
    return "stable"


def dominant_emotion(window: List[Dict[str, Any]]) -> str:
    from collections import Counter

    emotions = [str(s.get("emotion") or "").strip().lower() for s in window if s.get("emotion")]
    if not emotions:
        return "neutral"
    return Counter(emotions).most_common(1)[0][0]


def analyze_trends(window: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not window:
        return {
            "trend_direction": "stable",
            "stress_trend": "stable",
            "burnout_risk": "low",
            "emotional_stability": 50,
            "recovery_score": 0,
            "wellness_delta": 0,
            "dominant_emotion": "neutral",
            "trend_summary": ["Not enough history for trend analysis"],
        }

    # wellness delta (recent - older)
    wellness_scores = [s.get("wellness_score") for s in window if s.get("wellness_score") is not None]
    if len(wellness_scores) >= 2:
        half = len(wellness_scores) // 2
        first_avg = sum(wellness_scores[:half]) / max(1, half)
        second_avg = sum(wellness_scores[half:]) / max(1, len(wellness_scores) - half)
        wellness_delta = int(round(second_avg - first_avg))
    else:
        wellness_delta = 0

    trend_direction = _trend_direction(wellness_delta)
    stress_trend = _stress_direction([s.get("stress_level") for s in window])

    burnout_risk, burnout_signals, burnout_score = detect_burnout(window)
    emotional_stability, volatility, consistency, recovery_rate = compute_stability(window)
    recovery_score, recovery_improvements, recovery_regressions = compute_recovery(window)

    dom = dominant_emotion(window)

    # Build human-readable summary bullets
    trend_summary: List[str] = []
    if stress_trend == "decreasing":
        trend_summary.append("Stress levels decreased over recent sessions")
    elif stress_trend == "increasing":
        trend_summary.append("Stress levels increased over recent sessions")
    else:
        trend_summary.append("Stress levels remained relatively stable")

    # Sleep consistency / improvement when available
    sleep_vals = [s.get("sleep_hours") for s in window if s.get("sleep_hours") is not None]
    if len(sleep_vals) >= 4:
        half = len(sleep_vals) // 2
        first_avg = sum(sleep_vals[:half]) / max(1, half)
        second_avg = sum(sleep_vals[half:]) / max(1, len(sleep_vals) - half)
        if second_avg > first_avg:
            trend_summary.append("Sleep consistency improved")
        elif second_avg < first_avg:
            trend_summary.append("Sleep consistency worsened")

    if emotional_stability >= 70:
        trend_summary.append("Emotional stability increased")
    elif emotional_stability <= 45:
        trend_summary.append("Emotional volatility is elevated")

    if not trend_summary:
        trend_summary = ["Trend analysis unavailable due to missing data"]

    return {
        "trend_direction": trend_direction,
        "stress_trend": stress_trend,
        "burnout_risk": burnout_risk,
        "emotional_stability": int(emotional_stability),
        "recovery_score": int(recovery_score),
        "wellness_delta": int(wellness_delta),
        "dominant_emotion": dom,
        "trend_summary": trend_summary,
        # Internal extra fields (safe additive)
        "_debug": {
            "burnout_score": burnout_score,
            "burnout_signals": burnout_signals,
            "volatility": volatility,
            "consistency": consistency,
            "recovery_rate": recovery_rate,
        },
    }

