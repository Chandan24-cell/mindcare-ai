from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.copilot.empathy_engine import empathy_style
from backend.copilot.intent_detector import detect_intent
from backend.copilot.response_planner import plan_response


class WellnessCopilot:
    """Deterministic wellness copilot readiness layer."""

    def __init__(self) -> None:
        pass

    def build_copilot_context(
        self,
        *,
        user_text: str,
        escalation_analysis: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        personalization: Dict[str, Any],
    ) -> Dict[str, Any]:
        escalation_level = str(escalation_analysis.get("escalation_level") or "low")
        empathy = empathy_style(escalation_level=escalation_level)
        intent = detect_intent(user_text)

        response = plan_response(
            intent=intent.get("intent") or "general_wellness",
            empathy=empathy,
            alerts=alerts,
            personalization=personalization,
        )

        return {
            "intent": intent.get("intent"),
            "empathy": empathy,
            "response_plan": response,
        }

