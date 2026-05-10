from __future__ import annotations

from typing import Any, Dict, List, Optional


def simulate_intervention_effect(*, window: List[Dict[str, Any]], intervention: Optional[str] = None) -> Dict[str, Any]:
    # Deterministic projection; does not depend on intervention text heavily.
    if not window:
        return {
            "projected_effectiveness": 0.72,
            "recovery_acceleration": 0.18,
            "emotional_stabilization_impact": 0.22,
            "stress_reduction_likelihood": 0.7,
        }

    intervention = intervention or "mindfulness_break"
    stresses = [str(s.get("stress_level") or "").lower() for s in window]
    high_ratio = sum(1 for s in stresses if s in {"high", "critical"}) / max(1, len(stresses))

    effectiveness = max(0.25, min(0.9, 0.78 - 0.25 * high_ratio))
    recovery_acc = max(0.05, min(0.5, 0.12 + 0.25 * (1.0 - high_ratio)))

    return {
        "projected_effectiveness": round(float(effectiveness), 2),
        "recovery_acceleration": round(float(recovery_acc), 2),
        "emotional_stabilization_impact": round(float(0.15 + 0.2 * (1.0 - high_ratio)), 2),
        "stress_reduction_likelihood": round(float(0.65 + 0.2 * (1.0 - high_ratio)), 2),
    }

