from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image

from backend.ml.affective.face_mesh_engine import estimate_face_mesh_metrics


def analyze_face_landmarks(pil_image: Image.Image) -> Dict[str, Any]:
    """High-level face-landmark analysis entrypoint.

    Returns the standardized face_landmark_analysis payload.
    """

    metrics = estimate_face_mesh_metrics(pil_image)
    return {
        "eye_openness": metrics.eye_openness,
        "mouth_tension": metrics.mouth_tension,
        "brow_stress": metrics.brow_stress,
        "head_stability": metrics.head_tilt,
        "facial_symmetry": metrics.facial_symmetry,
        "landmark_confidence": metrics.landmark_confidence,
    }

