from __future__ import annotations

from typing import Any, Dict, List

from backend.realtime.anomaly_detector import detect_anomaly
from backend.realtime.emotional_drift_engine import analyze_emotional_drift
from backend.realtime.escalation_engine import detect_escalation
from backend.realtime.risk_forecasting import forecast_risk
from backend.realtime.cognitive_load_engine import estimate_cognitive_load
from backend.realtime.fatigue_detector import detect_fatigue
from backend.realtime.alert_engine import build_alerts


def realtime_monitoring(*, trend_window: List[Dict[str, Any]], burnout_analysis: Dict[str, Any]) -> Dict[str, Any]:
    drift = analyze_emotional_drift(trend_window)
    anomaly = detect_anomaly(trend_window)

    escalation = detect_escalation(
        trend_window=trend_window,
        burnout_analysis=burnout_analysis,
        drift_analysis=drift,
    )

    risk = forecast_risk(
        trend_window=trend_window,
        emotional_drift=drift,
        burnout_analysis=burnout_analysis,
    )

    cognitive = estimate_cognitive_load(trend_window=trend_window)
    fatigue = detect_fatigue(trend_window=trend_window)

    # stress velocity proxy
    def _stress_num(x: Any) -> float:
        s = str(x or "").lower()
        return {"low": 0.15, "medium": 0.5, "high": 0.82, "critical": 0.95}.get(s, 0.5)

    if len(trend_window) >= 10:
        last5 = trend_window[-5:]
        prev5 = trend_window[-10:-5]
        st_last = sum(_stress_num(s.get("stress_level")) for s in last5) / max(1, len(last5))
        st_prev = sum(_stress_num(s.get("stress_level")) for s in prev5) / max(1, len(prev5))
        stress_velocity = max(0.0, min(1.0, st_last - st_prev + 0.15))
    else:
        stress_velocity = 0.18

    realtime_risk = 0.25
    burnout_risk = str((burnout_analysis or {}).get("burnout_risk") or "low").lower()
    if burnout_risk in {"high", "critical"}:
        realtime_risk += 0.35
    realtime_risk += 0.25 * min(1.0, float(drift.get("drift_score") or 0.0))
    if anomaly.get("anomaly_detected"):
        realtime_risk += 0.15

    realtime_risk = max(0.0, min(1.0, realtime_risk))

    current_state = "stable"
    if escalation.get("escalation_detected"):
        current_state = "elevating_risk"
    elif realtime_risk >= 0.55:
        current_state = "unstable"

    alerts = build_alerts(
        escalation=escalation,
        anomaly=anomaly,
        drift=drift,
        burnout_analysis=burnout_analysis,
    )

    return {
        "current_state": current_state,
        "stress_velocity": round(float(stress_velocity), 2),
        "emotional_drift": round(float(drift.get("drift_score") or 0.0), 2),
        "realtime_risk_score": round(float(realtime_risk), 2),
        "monitoring_status": "active",
        "anomaly_analysis": anomaly,
        "emotional_drift": drift,
        "risk_forecast": risk,
        "escalation_analysis": escalation,
        "cognitive_load": cognitive,
        "fatigue_analysis": fatigue,
        "alerts": alerts,
    }

