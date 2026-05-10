from __future__ import annotations

from typing import Any, Dict

from backend.ml.affective.emotion_memory_buffer import EmotionMemoryBuffer
from backend.ml.affective.emotion_stabilizer import stabilize_emotion


def run_temporal_emotion_engine(
    *,
    current_emotion: str,
    current_confidence: float,
    memory: EmotionMemoryBuffer,
) -> Dict[str, Any]:
    """Run temporal stabilization."""

    return stabilize_emotion(
        current_emotion=current_emotion,
        current_confidence=current_confidence,
        memory=memory,
    )

