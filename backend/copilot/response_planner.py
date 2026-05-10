from __future__ import annotations

from typing import Any, Dict, List, Optional


def plan_response(*, intent: str, empathy: Dict[str, Any], alerts: List[Dict[str, Any]], personalization: Dict[str, Any]) -> Dict[str, Any]:
    """Plan a deterministic wellness copilot response payload."""

    tone = empathy.get("tone") or "supportive"

    best_alert = alerts[0] if alerts else None
    if best_alert:
        message = best_alert.get("message")
        action = best_alert.get("action")
    else:
        message = "You are making progress—let's keep your recovery steady."
        action = "Try a short breathing routine (3 minutes) and a brief hydration break."

    interventions = personalization.get("preferred_interventions") or []
    if interventions:
        action = f"{action} Suggested: {interventions[0]}."

    strategy = empathy.get("response_strategy") or "encouragement"

    return {
        "tone": tone,
        "intent": intent,
        "strategy": strategy,
        "message": message,
        "action": action,
    }

