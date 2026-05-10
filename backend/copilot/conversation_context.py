from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ConversationContext:
    user_text: str
    recent_emotion: Optional[str] = None
    recent_stress_level: Optional[str] = None
    burnout_risk: Optional[str] = None
    preferred_interventions: Optional[list[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_text": self.user_text,
            "recent_emotion": self.recent_emotion,
            "recent_stress_level": self.recent_stress_level,
            "burnout_risk": self.burnout_risk,
            "preferred_interventions": self.preferred_interventions or [],
        }

