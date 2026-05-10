from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


TrendDirection = Literal["improving", "worsening", "stable"]


@dataclass(frozen=True)
class TrendAnalyticsResult:
    trend_direction: str
    stress_trend: str
    burnout_risk: str
    emotional_stability: int
    recovery_score: int
    wellness_delta: int
    dominant_emotion: str
    trend_summary: List[str]


@dataclass(frozen=True)
class BurnoutAnalysisResult:
    burnout_risk: str
    signals: Dict[str, Any]
    score: int


@dataclass(frozen=True)
class StabilityAnalysisResult:
    emotional_stability: int
    volatility: int
    consistency: int
    recovery_rate: int


@dataclass(frozen=True)
class RecoveryAnalysisResult:
    recovery_score: int
    improvements: List[str]
    regressions: List[str]


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    timestamp: str
    emotion: str
    stress_level: str
    confidence: float
    wellness_score: int
    severity: str
    sleep_hours: Optional[float] = None
    heart_rate: Optional[float] = None
    hrv: Optional[float] = None
    activity_level: Optional[int] = None
    source_modalities: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "emotion": self.emotion,
            "stress_level": self.stress_level,
            "confidence": self.confidence,
            "wellness_score": self.wellness_score,
            "severity": self.severity,
            "sleep_hours": self.sleep_hours,
            "heart_rate": self.heart_rate,
            "hrv": self.hrv,
            "activity_level": self.activity_level,
            "source_modalities": self.source_modalities or [],
        }


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

