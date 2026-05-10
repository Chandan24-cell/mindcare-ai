from __future__ import annotations

from typing import Any, Dict, List


def build_behavioral_graph(*, window: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Build simple deterministic graph edges from stress transitions.
    transitions: Dict[str, Dict[str, int]] = {}
    emotions = [str(x.get("emotion") or "").lower() for x in window if x.get("emotion")]

    for i in range(1, len(emotions)):
        a = emotions[i - 1]
        b = emotions[i]
        transitions.setdefault(a, {})
        transitions[a][b] = transitions[a].get(b, 0) + 1

    return {
        "emotional_transition_graph": transitions,
        "stressor_relationships": {
            "notes": "Full stressor dependency mapping reserved for future research pipeline.",
        },
    }

