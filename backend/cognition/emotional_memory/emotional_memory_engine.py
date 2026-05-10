from __future__ import annotations

from typing import Any, Dict, List


def build_emotional_memory(*, window: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Lightweight deterministic memory summary.
    emotions = [str(x.get("emotion") or "").lower() for x in window if x.get("emotion")]
    recurrence: Dict[str, int] = {}
    for e in emotions:
        recurrence[e] = recurrence.get(e, 0) + 1

    top_trigger = None
    if emotions:
        # Use most frequent negative emotion as trigger proxy.
        negative = [e for e in emotions if e in {"sad", "angry", "fear", "disgust"}]
        if negative:
            top_trigger = max(set(negative), key=lambda k: recurrence.get(k, 0))

    return {
        "emotional_recurrence": recurrence,
        "top_trigger": top_trigger,
        "recovery_memory": {
            "notes": "Recovery memory available in future persistence layer.",
        },
    }

