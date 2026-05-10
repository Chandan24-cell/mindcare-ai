from __future__ import annotations

from typing import Any, Dict, List, Optional


def learn_intervention(*, window: List[Dict[str, Any]], legacy_suggestions: Optional[List[str]] = None) -> Dict[str, Any]:
    # Deterministic placeholder: choose the first legacy suggestion as effective.
    suggestions = legacy_suggestions or []
    most_effective = suggestions[0] if suggestions else "mindfulness_break"

    # effectiveness score based on presence of high/critical stress history
    stresses = [str(x.get("stress_level") or "").lower() for x in window if x.get("stress_level")]
    high_ratio = sum(1 for s in stresses if s in {"high", "critical"}) / max(1, len(stresses))

    effectiveness = max(0.35, min(0.9, 0.8 - 0.25 * high_ratio))
    adaptation_level = "personalized" if effectiveness >= 0.6 else "generalized"

    return {
        "most_effective_intervention": most_effective,
        "effectiveness_score": round(float(effectiveness), 2),
        "adaptation_level": adaptation_level,
    }

