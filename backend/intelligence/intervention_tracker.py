from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _default_intervention_stats() -> Dict[str, Dict[str, Any]]:
    # Each intervention: {uses, successes}
    return {
        "breathing": {"uses": 0, "successes": 0},
        "music": {"uses": 0, "successes": 0},
        "walk": {"uses": 0, "successes": 0},
        "stretch": {"uses": 0, "successes": 0},
    }


def analyze_intervention_effectiveness(
    *,
    intervention_effectiveness: Optional[Dict[str, Dict[str, Any]]] = None,
    severity: str,
    recovery_score: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute which intervention seems most effective.

    This is conservative and safe for missing data.
    """

    stats = intervention_effectiveness or _default_intervention_stats()

    best_name: Optional[str] = None
    best_score = -1.0
    best_success_rate = 0.0

    for name, st in stats.items():
        uses = float(st.get("uses") or 0)
        successes = float(st.get("successes") or 0)
        if uses <= 0:
            success_rate = 0.0
        else:
            success_rate = successes / uses
        # effectiveness_score: incorporate success rate + recovery signal
        rec = float(recovery_score or 0) / 100.0
        effectiveness = 0.7 * success_rate + 0.3 * rec
        if effectiveness > best_score:
            best_score = effectiveness
            best_name = name
            best_success_rate = success_rate

    best_name = best_name or "breathing"

    # recommended frequency based on severity
    sev = str(severity or "moderate").lower().strip()
    if sev in {"critical", "high"}:
        freq = "daily"
    elif sev == "moderate":
        freq = "every-other-day"
    else:
        freq = "weekly"

    return {
        "most_effective_intervention": best_name,
        "effectiveness_score": round(float(best_score), 2),
        "recommended_frequency": freq,
        "historical_success_rate": round(float(best_success_rate), 2),
    }

