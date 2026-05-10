from __future__ import annotations

from typing import Any, Dict, List


def detect_fatigue(*, trend_window: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trend_window:
        return {
            "fatigue_detected": False,
            "fatigue_level": "low",
            "fatigue_confidence": 0.3,
        }

    wellness = [s.get("wellness_score") for s in trend_window if s.get("wellness_score") is not None]
    sleep = [s.get("sleep_hours") for s in trend_window if s.get("sleep_hours") is not None]
    emotions = [str(s.get("emotion") or "").lower() for s in trend_window if s.get("emotion")]

    low_wellness = sum(1 for w in wellness if w < 50) / max(1, len(wellness)) if wellness else 0.0
    low_sleep = sum(1 for h in sleep if h < 5.0) / max(1, len(sleep)) if sleep else 0.0
    neg = sum(1 for e in emotions if e in {"sad", "angry", "fear", "disgust"}) / max(1, len(emotions)) if emotions else 0.0

    score = 0.4 * low_wellness + 0.3 * low_sleep + 0.3 * neg
    score = max(0.0, min(1.0, score))

    fatigue_detected = score >= 0.45

    fatigue_level = "low"
    if score >= 0.55:
        fatigue_level = "moderate"
    if score >= 0.75:
        fatigue_level = "high"

    fatigue_confidence = round(0.35 + 0.6 * score, 2)

    return {
        "fatigue_detected": fatigue_detected,
        "fatigue_level": fatigue_level,
        "fatigue_confidence": fatigue_confidence,
    }

