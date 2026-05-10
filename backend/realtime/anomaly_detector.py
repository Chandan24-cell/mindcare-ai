from __future__ import annotations

from typing import Any, Dict, List, Optional


_NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}


def _stress_numeric(stress_level: Optional[str]) -> float:
    s = str(stress_level or "").lower().strip()
    return {"low": 0.15, "medium": 0.5, "high": 0.82, "critical": 0.95}.get(
        s, 0.5
    )


def _emotion_negative(emotion: Optional[str]) -> bool:
    return str(emotion or "").lower().strip() in _NEGATIVE_EMOTIONS


def detect_anomaly(window: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not window:
        return {
            "anomaly_detected": False,
            "anomaly_type": "none",
            "severity": "low",
            "confidence": 0.0,
            "details": [],
        }

    last = window[-1]
    prev = window[-2] if len(window) >= 2 else None
    prev2 = window[-3] if len(window) >= 3 else None

    details: List[str] = []

    # Sudden emotion spikes (negative shift)
    if prev2 is not None and prev is not None:
        e0 = _emotion_negative(prev2.get("emotion"))
        e1 = _emotion_negative(prev.get("emotion"))
        e2 = _emotion_negative(last.get("emotion"))
        if not e0 and not e1 and e2:
            details.append("Stress/emotion shifted into negative category")

    # Abnormal stress transitions
    if prev is not None:
        st_prev = _stress_numeric(prev.get("stress_level"))
        st_last = _stress_numeric(last.get("stress_level"))
        if st_last - st_prev >= 0.35:
            details.append("Stress increased rapidly within 3 sessions")

    # HRV baseline drop
    hrv_vals = [s.get("hrv") for s in window if s.get("hrv") is not None]
    if len(hrv_vals) >= 2:
        if float(hrv_vals[-1]) < (sum(hrv_vals[-2:]) / 2.0):
            details.append("HRV dropped below baseline")

    anomaly_detected = len(details) > 0
    if not anomaly_detected:
        return {
            "anomaly_detected": False,
            "anomaly_type": "none",
            "severity": "low",
            "confidence": 0.2,
            "details": [],
        }

    anomaly_type = "stress_spike" if any("Stress increased" in d for d in details) else "emotion_spike"

    severity_score = min(1.0, 0.45 + 0.25 * len(details))
    severity = "low"
    if severity_score >= 0.65:
        severity = "medium"
    if severity_score >= 0.85:
        severity = "high"

    confidence = min(0.95, 0.45 + 0.15 * len(details))

    return {
        "anomaly_detected": True,
        "anomaly_type": anomaly_type,
        "severity": severity,
        "confidence": round(confidence, 2),
        "details": details,
    }

