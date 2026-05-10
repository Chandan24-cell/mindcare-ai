from __future__ import annotations

from typing import Any, Dict


def empathy_style(*, escalation_level: str, tone_preference: str | None = None) -> Dict[str, Any]:
    tone = "supportive" if str(escalation_level).lower() in {"high", "critical"} else "calm"
    if tone_preference:
        tone = tone_preference
    return {
        "tone": tone,
        "communication_style": "calm" if tone in {"supportive", "calm"} else "encouraging",
        "response_strategy": "encouragement" if tone in {"supportive", "calm"} else "guidance",
    }

