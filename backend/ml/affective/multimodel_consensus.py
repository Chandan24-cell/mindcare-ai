from __future__ import annotations

from typing import Any, Dict

from backend.ml.affective.multimodel_consensus import build_model_consensus as _b


def model_consensus(
    *,
    vit_emotion: str,
    vit_confidence: float,
    stabilized_emotion: str,
    calibrated_confidence: float,
    facial_signal_confidence: float | None = None,
) -> Dict[str, Any]:
    return _b(
        vit_emotion=vit_emotion,
        vit_confidence=vit_confidence,
        stabilized_emotion=stabilized_emotion,
        calibrated_confidence=calibrated_confidence,
        facial_signal_confidence=facial_signal_confidence,
    )

