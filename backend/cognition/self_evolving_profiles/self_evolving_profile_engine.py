from __future__ import annotations

from typing import Any, Dict, List


def compute_adaptive_personalization(*, window: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Deterministic personalization summary.
    emotions = [str(x.get("emotion") or "").lower() for x in window if x.get("emotion")]
    neg_count = sum(1 for e in emotions if e in {"sad", "angry", "fear", "disgust"})
    if not emotions:
        dominant = "neutral"
    else:
        dominant = max(set(emotions), key=lambda k: emotions.count(k))

    adaptation_score = max(0.0, min(1.0, 0.45 + 0.25 * (1.0 - neg_count / max(1, len(emotions)))))

    return {
        "dominant_emotion": dominant,
        "adaptation_score": round(float(adaptation_score), 2),
        "notes": "Future persistence can update this profile across sessions.",
    }

