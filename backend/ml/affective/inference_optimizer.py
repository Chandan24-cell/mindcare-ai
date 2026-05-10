from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from PIL import Image


@dataclass(frozen=True)
class InferenceOptimizationPlan:
    """Optimization hints for affective inference."""

    use_downscale: bool
    downscale_max_side: int
    enable_mediapipe_if_available: bool


def build_inference_plan(*, pil_image: Image.Image) -> InferenceOptimizationPlan:
    w, h = pil_image.size
    max_side = max(w, h)
    use_downscale = max_side > 800
    return InferenceOptimizationPlan(
        use_downscale=use_downscale,
        downscale_max_side=640,
        enable_mediapipe_if_available=True,
    )


def maybe_downscale(pil_image: Image.Image, *, max_side: int) -> Image.Image:
    w, h = pil_image.size
    if max(w, h) <= max_side:
        return pil_image
    ratio = max_side / float(max(w, h))
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))
    return pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)

