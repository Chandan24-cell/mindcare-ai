from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _stress_val(stress_level: Optional[str]) -> float:
    s = str(stress_level or "").lower()
    return {"low": 0.15, "medium": 0.5, "high": 0.82, "critical": 0.95}.get(s, 0.5)


def compute_recovery(window: List[Dict[str, Any]]) -> Tuple[int, List[str], List[str]]:
    if not window:
        return 0, [], []

    # Compare first half vs second half for stress + wellness when available
    stresses = [_stress_val(s.get("stress_level")) for s in window]
    wellness = [s.get("wellness_score") for s in window if s.get("wellness_score") is not None]

    half = max(1, len(window) // 2)
    first_stress = sum(stresses[:half]) / len(stresses[:half])
    second_stress = sum(stresses[half:]) / len(stresses[half:]) if len(stresses) > half else first_stress

    # Lower stress => improvement
    stress_improvement = max(0.0, min(1.0, (first_stress - second_stress) / 0.85))

    wellness_improvement = 0.0
    if len(wellness) >= 2:
        whalf = max(1, len(wellness) // 2)
        first_w = sum(wellness[:whalf]) / len(wellness[:whalf])
        second_w = sum(wellness[whalf:]) / len(wellness[whalf:]) if len(wellness) > whalf else first_w
        wellness_improvement = max(0.0, min(1.0, (second_w - first_w) / 40.0))

    score01 = 0.60 * stress_improvement + 0.40 * wellness_improvement
    score = int(round(score01 * 100))

    improvements: List[str] = []
    regressions: List[str] = []

    if stress_improvement > 0.35:
        improvements.append("Stress levels decreased over recent sessions")
    if wellness_improvement > 0.2:
        improvements.append("Wellness score improved")

    if stress_improvement < 0.15:
        regressions.append("Stress reduction not clearly observed")

    return score, improvements, regressions

