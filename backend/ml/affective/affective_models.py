from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FacialLandmarkMetrics:
    """Structured outputs for facial-landmark-derived metrics."""

    eye_openness: float | None
    mouth_tension: float | None
    brow_stress: float | None
    head_tilt: float | None
    facial_symmetry: float | None
    landmark_confidence: float | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eye_openness": self.eye_openness,
            "mouth_tension": self.mouth_tension,
            "brow_stress": self.brow_stress,
            "head_tilt": self.head_tilt,
            "facial_symmetry": self.facial_symmetry,
            "landmark_confidence": self.landmark_confidence,
        }


@dataclass(frozen=True)
class MicroExpressionMetrics:
    micro_expression_detected: bool
    micro_expression_score: float
    hidden_stress_probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "micro_expression_detected": self.micro_expression_detected,
            "micro_expression_score": self.micro_expression_score,
            "hidden_stress_probability": self.hidden_stress_probability,
        }


@dataclass(frozen=True)
class TemporalStabilization:
    stabilized_emotion: str
    stability_score: float
    emotion_variance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stabilized_emotion": self.stabilized_emotion,
            "stability_score": self.stability_score,
            "emotion_variance": self.emotion_variance,
        }


@dataclass(frozen=True)
class ConfidenceCalibration:
    raw_confidence: float
    calibrated_confidence: float
    uncertainty_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "uncertainty_score": self.uncertainty_score,
        }


@dataclass(frozen=True)
class FacialSignals:
    fatigue_signal: float
    engagement_signal: float
    cognitive_strain: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fatigue_signal": self.fatigue_signal,
            "engagement_signal": self.engagement_signal,
            "cognitive_strain": self.cognitive_strain,
        }


@dataclass(frozen=True)
class ModelConsensus:
    consensus_emotion: str
    consensus_confidence: float
    agreement_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus_emotion": self.consensus_emotion,
            "consensus_confidence": self.consensus_confidence,
            "agreement_score": self.agreement_score,
        }


@dataclass(frozen=True)
class ReliabilityEstimation:
    reliability_score: float
    quality_flags: list[str]
    low_quality_detected: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reliability_score": self.reliability_score,
            "quality_flags": self.quality_flags,
            "low_quality_detected": self.low_quality_detected,
        }

