from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RealtimeMonitoringOutput:
    current_state: str
    stress_velocity: float
    emotional_drift: float
    realtime_risk_score: float
    monitoring_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state,
            "stress_velocity": self.stress_velocity,
            "emotional_drift": self.emotional_drift,
            "realtime_risk_score": self.realtime_risk_score,
            "monitoring_status": self.monitoring_status,
        }


@dataclass(frozen=True)
class AnomalyAnalysisOutput:
    anomaly_detected: bool
    anomaly_type: str
    severity: str
    confidence: float
    details: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_detected": self.anomaly_detected,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "details": self.details,
        }


@dataclass(frozen=True)
class DriftOutput:
    drift_direction: str
    drift_score: float
    baseline_divergence: float
    stability_impact: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_direction": self.drift_direction,
            "drift_score": self.drift_score,
            "baseline_divergence": self.baseline_divergence,
            "stability_impact": self.stability_impact,
        }


@dataclass(frozen=True)
class EscalationOutput:
    escalation_detected: bool
    escalation_level: str
    escalation_reason: str
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "escalation_detected": self.escalation_detected,
            "escalation_level": self.escalation_level,
            "escalation_reason": self.escalation_reason,
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True)
class RiskForecastOutput:
    future_risk_level: str
    forecast_confidence: float
    predicted_direction: str
    estimated_recovery_sessions: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "future_risk_level": self.future_risk_level,
            "forecast_confidence": self.forecast_confidence,
            "predicted_direction": self.predicted_direction,
            "estimated_recovery_sessions": self.estimated_recovery_sessions,
        }


@dataclass(frozen=True)
class CognitiveLoadOutput:
    cognitive_load: str
    load_score: float
    mental_fatigue_risk: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cognitive_load": self.cognitive_load,
            "load_score": self.load_score,
            "mental_fatigue_risk": self.mental_fatigue_risk,
        }


@dataclass(frozen=True)
class FatigueOutput:
    fatigue_detected: bool
    fatigue_level: str
    fatigue_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fatigue_detected": self.fatigue_detected,
            "fatigue_level": self.fatigue_level,
            "fatigue_confidence": self.fatigue_confidence,
        }

