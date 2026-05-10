from __future__ import annotations

from typing import Any, Dict

from backend.ml.affective.emotion_memory_buffer import EmotionMemoryBuffer


def stabilize_emotion(
    *,
    current_emotion: str,
    current_confidence: float,
    memory: EmotionMemoryBuffer,
) -> Dict[str, Any]:
    """Temporal emotion stabilization.

    Deterministic weighted vote over a rolling buffer.
    """

    try:
        memory.append(emotion=current_emotion, confidence=current_confidence)
    except Exception:
        pass

    items = memory.items()
    if not items:
        return {
            "stabilized_emotion": current_emotion,
            "stability_score": 0.5,
            "emotion_variance": 0.0,
        }

    weights: Dict[str, float] = {}
    for it in items:
        weights[it.emotion] = weights.get(it.emotion, 0.0) + float(it.confidence)

    stabilized_emotion = max(weights.items(), key=lambda kv: kv[1])[0]

    distinct = len(weights)
    emotion_variance = min(1.0, 0.05 + 0.25 * max(0, distinct - 1))

    total = sum(weights.values()) or 1.0
    dominant = weights.get(stabilized_emotion, 0.0) / total
    stability_score = max(0.0, min(1.0, 0.2 + 0.8 * dominant - 0.15 * emotion_variance))

    return {
        "stabilized_emotion": stabilized_emotion,
        "stability_score": round(float(stability_score), 2),
        "emotion_variance": round(float(emotion_variance), 2),
    }

