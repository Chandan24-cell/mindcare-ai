from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_adaptive_recommendations(
    *,
    legacy_suggestions: List[str],
    personalization: Dict[str, Any],
    severity: str,
    burnout_risk: Optional[str] = None,
    recovery_score: Optional[int] = None,
) -> List[str]:
    """Return adapted recommendations as strings.

    This function is conservative: it preserves legacy suggestions and only
    reorders/annotates them based on personalization signals.
    """

    preferred = [
        str(x).lower().strip()
        for x in (personalization.get("preferred_interventions") or [])
    ]
    dominant_trigger = str(personalization.get("dominant_stress_trigger") or "")
    burnout = str(burnout_risk or "").lower().strip()

    scored: List[tuple[int, int, str]] = []
    for idx, s in enumerate(legacy_suggestions or []):
        text = str(s)
        low = text.lower()
        score = 0
        for p in preferred:
            if p and p in low:
                score += 2
        if dominant_trigger and dominant_trigger in low:
            score += 1
        if burnout in {"high", "critical"} and "breath" in low:
            score += 1
        if str(severity or "").lower().strip() in {"critical", "high"} and "sleep" in low:
            score += 1

        scored.append((score, idx, text))

    scored.sort(key=lambda t: (-t[0], t[1]))
    ordered = [t[2] for t in scored]

    if not ordered:
        ordered = legacy_suggestions[:6] if legacy_suggestions else [
            "Practice deep breathing",
            "Try calming music",
            "Take a short mindful walk",
        ]

    # Limit size to keep UI stable.
    recs = ordered[:6]

    # Add one personalization qualifier if possible.
    qualifier = None
    if preferred:
        qualifier = f"Personalized to your preferred interventions: {', '.join(preferred[:3])}."
    if dominant_trigger and not qualifier:
        qualifier = f"Targeting your dominant stress trigger: {dominant_trigger}."

    if qualifier:
        recs.append(qualifier)

    return recs

