"""
Utility validators for input images and predictions.

These helpers keep the main inference code clean and focused on the
business logic while still enforcing basic quality checks.
"""

from typing import Tuple
import numpy as np
from PIL import Image


def validate_image_size(image: Image.Image, min_side: int = 64) -> Tuple[bool, str]:
    """
    Ensure the incoming image is large enough for reliable face detection.
    """
    w, h = image.size
    if min(w, h) < min_side:
        return False, f"Image too small ({w}x{h}). Please provide a larger image."
    return True, ""


def validate_brightness(image: Image.Image, min_brightness: float = 25.0) -> Tuple[bool, str]:
    """
    Basic brightness check to avoid running inference on nearly dark frames.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    mean_brightness = float(gray.mean())
    if mean_brightness < min_brightness:
        return False, "Image too dark. Improve lighting and try again."
    return True, ""
