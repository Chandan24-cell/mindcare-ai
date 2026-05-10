from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ExplainabilityOutput:
    primary_factor: str
    secondary_factor: Optional[str]
    supporting_factors: List[str]
    confidence_reason: str
    modality_agreement: float
    explanation_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_factor": self.primary_factor,
            "secondary_factor": self.secondary_factor,
            "supporting_factors": self.supporting_factors,
            "confidence_reason": self.confidence_reason,
            "modality_agreement": self.modality_agreement,
            "explanation_summary": self.explanation_summary,
        }


@dataclass(frozen=True)
class ModalityReliabilityOutput:
    facial_reliability: float
    sensor_reliability: float
    manual_input_reliability: float
    overall_modality_agreement: float
    dominant_modality: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facial_reliability": self.facial_reliability,
            "sensor_reliability": self.sensor_reliability,
            "manual_input_reliability": self.manual_input_reliability,
            "overall_modality_agreement": self.overall_modality_agreement,
            "dominant_modality": self.dominant_modality,
        }


@dataclass(frozen=True)
class BehavioralProfileOutput:
    behavioral_type: str
    emotional_volatility: float
    stress_resilience: float
    wellness_consistency: float
    risk_profile: str
    adaptation_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavioral_type": self.behavioral_type,
            "emotional_volatility": self.emotional_volatility,
            "stress_resilience": self.stress_resilience,
            "wellness_consistency": self.wellness_consistency,
            "risk_profile": self.risk_profile,
            "adaptation_score": self.adaptation_score,
        }


@dataclass(frozen=True)
class PersonalizationOutput:
    baseline_emotion: str
    dominant_stress_trigger: str
    recovery_speed: str
    preferred_interventions: List[str]
    highest_risk_period: str
    behavioral_pattern: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_emotion": self.baseline_emotion,
            "dominant_stress_trigger": self.dominant_stress_trigger,
            "recovery_speed": self.recovery_speed,
            "preferred_interventions": self.preferred_interventions,
            "highest_risk_period": self.highest_risk_period,
            "behavioral_pattern": self.behavioral_pattern,
        }


@dataclass(frozen=True)
class InterventionAnalysisOutput:
    most_effective_intervention: str
    effectiveness_score: float
    recommended_frequency: str
    historical_success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "most_effective_intervention": self.most_effective_intervention,
            "effectiveness_score": self.effectiveness_score,
            "recommended_frequency": self.recommended_frequency,
            "historical_success_rate": self.historical_success_rate,
        }


@dataclass(frozen=True)
class UserProfileState:
    """Internal structure for persisted user profiles."""

    user_id: str
    baseline_emotion: str
    preferred_interventions: List[str]
    stress_trigger_histogram: Dict[str, float]
    recovery_speed_score: float
    behavioral_pattern: str
    highest_risk_period: str

    # rolling metrics
    emotional_stability_history: List[float]
    stress_level_history: List[str]
    wellness_score_history: List[int]

    # modality reliability rolling
    modality_reliability_history: List[Dict[str, Any]]

    # intervention effectiveness
    intervention_effectiveness: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "baseline_emotion": self.baseline_emotion,
            "preferred_interventions": self.preferred_interventions,
            "stress_trigger_histogram": self.stress_trigger_histogram,
            "recovery_speed_score": self.recovery_speed_score,
            "behavioral_pattern": self.behavioral_pattern,
            "highest_risk_period": self.highest_risk_period,
            "emotional_stability_history": self.emotional_stability_history,
            "stress_level_history": self.stress_level_history,
            "wellness_score_history": self.wellness_score_history,
            "modality_reliability_history": self.modality_reliability_history,
            "intervention_effectiveness": self.intervention_effectiveness,
        }

    @staticmethod
    def default(user_id: str) -> "UserProfileState":
        return UserProfileState(
            user_id=user_id,
            baseline_emotion="neutral",
            preferred_interventions=["breathing", "music"],
            stress_trigger_histogram={},
            recovery_speed_score=0.5,
            behavioral_pattern="stress often increases when sleep is low",
            highest_risk_period="unknown",
            emotional_stability_history=[],
            stress_level_history=[],
            wellness_score_history=[],
            modality_reliability_history=[],
            intervention_effectiveness={},
        )

