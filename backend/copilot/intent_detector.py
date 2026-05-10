from __future__ import annotations

from typing import Any, Dict


def detect_intent(user_text: str) -> Dict[str, Any]:
    t = str(user_text or "").lower()

    if any(k in t for k in ["burnout", "exhaust", "tired", "fatigue"]):
        return {"intent": "burnout_concern"}
    if any(k in t for k in ["sleep", "insomnia", "late night", "tired"]):
        return {"intent": "recovery_planning"}
    if any(k in t for k in ["stress", "anxious", "worried"]):
        return {"intent": "stress_discussion"}
    if any(k in t for k in ["help", "what should i do", "recommend"]):
        return {"intent": "help_seeking"}

    return {"intent": "general_wellness"}

