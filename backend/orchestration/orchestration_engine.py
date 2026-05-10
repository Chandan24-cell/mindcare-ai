from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.cognition.temporal_reasoning.temporal_reasoning_engine import compute_temporal_reasoning
from backend.cognition.cognitive_state.cognitive_state_engine import compute_cognitive_state
from backend.cognition.emotional_memory.emotional_memory_engine import build_emotional_memory
from backend.cognition.behavioral_graph.behavioral_graph_engine import build_behavioral_graph
from backend.cognition.intervention_learning.intervention_learning_engine import learn_intervention
from backend.cognition.self_evolving_profiles.self_evolving_profile_engine import compute_adaptive_personalization
from backend.simulation.future_state_engine.future_state_engine import simulate_future_state


def orchestrate_research_intelligence(
    *,
    window: List[Dict[str, Any]],
    legacy_suggestions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    temporal = compute_temporal_reasoning(window=window)
    cognitive = compute_cognitive_state(window=window)
    emotional_memory = build_emotional_memory(window=window)
    behavioral_graph = build_behavioral_graph(window=window)

    intervention_learning = learn_intervention(
        window=window,
        legacy_suggestions=legacy_suggestions,
    )

    adaptive_personalization = compute_adaptive_personalization(window=window)
    future_state_simulation = simulate_future_state(window=window)

    orchestration_insights = {
        "orchestration_mode": "research_grade_temporal_cognition",
        "fusion_strategy": "deterministic_weighted_derivations",
        "inputs_window_size": len(window),
    }

    knowledge_graph_context = {
        "notes": "Knowledge graph reserved for future taxonomic enrichment.",
        "emotional_semantics": emotional_memory.get("emotional_recurrence", {}),
    }

    longitudinal_analysis = {
        "longitudinal_notes": "Longitudinal analysis pipeline reserved; current implementation derives trend proxies from existing window.",
        "trajectory_confidence": temporal.get("trajectory_confidence"),
    }

    return {
        "temporal_reasoning": temporal,
        "cognitive_state": cognitive,
        "future_state_simulation": future_state_simulation,
        "behavioral_graph_analysis": behavioral_graph,
        "intervention_learning": intervention_learning,
        "adaptive_personalization": adaptive_personalization,
        "orchestration_insights": orchestration_insights,
        "knowledge_graph_context": knowledge_graph_context,
        "longitudinal_analysis": longitudinal_analysis,
    }

