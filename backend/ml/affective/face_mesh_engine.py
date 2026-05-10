from __future__ import annotations

from typing import Dict, Any

from PIL import Image

from backend.ml.affective.face_mesh_engine import estimate_face_mesh_metrics


def run_face_mesh_engine(pil_image: Image.Image) -> Dict[str, Any]:
    """Wrapper for compatibility with the pipeline."""

    metrics = estimate_face_mesh_metrics(pil_image)
    return metrics.to_dict()

