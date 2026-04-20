# =============================================================================
# Inference Module
# =============================================================================
# This module handles all prediction/inference logic for the ML model.
# It provides functions for:
# - Real ML-based predictions using the ViT model
# - Mock predictions for demonstration/testing
# - Sensor data analysis
# - Manual input analysis
#
# Why Separate Inference Module:
# - Isolates ML logic from API routes
# - Easier to test inference independently
# - Allows swapping ML models without changing API
# - Centralizes prediction business logic
# =============================================================================

import random
import logging
from functools import lru_cache
from typing import Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image

# Set up logger
logger = logging.getLogger(__name__)

from backend.model_loader import device, load_vit_model
from transformers import ViTImageProcessor

from backend.face_detection import (
    detect_and_crop_face,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    FaceDetectionResult,
)
from backend.utils.validation import validate_image_size, validate_brightness


# =============================================================================
# Emotion to Stress Mapping
# =============================================================================
# Maps detected emotions to inferred stress levels.
# Based on psychological research on emotion-stress correlation.
# =============================================================================

EMOTION_TO_STRESS_MAP = {
    "happy": "low",
    "neutral": "medium",
    "sad": "medium",
    "angry": "high",
    "fear": "high",
    "disgust": "high",
    "surprise": "low"
}


# =============================================================================
# Cropped Face Validation
# =============================================================================
def _validate_cropped_face(cropped_face: Image.Image) -> bool:
    """
    Validate that the cropped region contains a real human face.

    This additional check prevents the model from running on false positives
    where the face detector returns a bounding box that does not actually
    contain a face (e.g., patterns on walls, similar shapes).

    Validation criteria:
    - Minimum dimensions (at least 32x32 pixels)
    - Sufficient variance (faces have texture; uniform areas are rejected)
    - Reasonable brightness (not too dark or too bright)

    Args:
        cropped_face: PIL Image containing the cropped face region

    Returns:
        True if the crop appears to be a valid face, False otherwise
    """
    try:
        # Convert to grayscale numpy array
        gray = np.array(cropped_face.convert('L'), dtype=np.uint8)

        # Check minimum size
        h, w = gray.shape
        if h < 32 or w < 32:
            logger.warning(f"Face crop too small: {w}x{h}")
            return False

        # Check variance - faces have texture variation (edges, shadows)
        std = float(gray.std())
        if std < 10:  # Very low variance suggests uniform region
            logger.warning(f"Face crop variance too low: {std:.2f}")
            return False

        # Check brightness extremes (overexposed/underexposed)
        mean = float(gray.mean())
        if mean < 20 or mean > 235:
            logger.warning(f"Face crop brightness extreme: {mean:.2f}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error validating face crop: {e}")
        return False


# =============================================================================
# Core helper: run ViT on a cropped face
# =============================================================================


@lru_cache(maxsize=1)
def _get_processor() -> ViTImageProcessor:
    return ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


def _run_vit_on_face(cropped_face: Image.Image) -> Tuple[str, str, float]:
    """
    Run the Vision Transformer on a pre-cropped face image.

    Keeping this logic in a small helper lets us reuse the same
    preprocessing and inference flow for both uploaded images and
    webcam frames without duplicating code.
    """
    model = load_vit_model()
    processor = _get_processor()

    # Preprocess for ViT (resize, normalize, tensor conversion)
    inputs = processor(images=cropped_face, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = logits.argmax(-1).item()
        confidence = probabilities[0][predicted_class].item()

    emotion = model.config.id2label[predicted_class]
    stress_level = EMOTION_TO_STRESS_MAP.get(emotion, "medium")

    # Sanity-check very low confidence predictions
    if confidence < 0.40:
        raise NoFaceDetectedError("Prediction confidence too low. Please try again with better lighting.")

    return emotion, stress_level, confidence


# =============================================================================
# Real ML Inference Functions
# =============================================================================

def predict_emotion_from_image(image: Image.Image) -> Tuple[str, str, float]:
    """
    Perform real ML inference on an image to detect emotion.

    # ----------------------------------------------------
    # PIPELINE OVERVIEW
    # ----------------------------------------------------
    # 1) FACE DETECTION: stop early if no face is visible.
    # 2) FACE CROP: isolate the main face so the model only
    #    sees relevant pixels.
    # 3) PREPROCESS: use the ViT processor (resize, normalize).
    # 4) INFERENCE: run the Vision Transformer.
    # 5) POST-PROCESS: map class -> emotion -> stress level.
    # ----------------------------------------------------

    Args:
        image: PIL Image object (RGB format)

    Returns:
        Tuple containing:
            - emotion: Detected emotion string (e.g., "happy", "sad")
            - stress_level: Inferred stress level ("low", "medium", "high")
            - confidence: Model confidence score (0.0 - 1.0)

    Raises:
        NoFaceDetectedError: if no face is found in the image.
    """
    # ----------------------------------------------------
    # STEP 0: Basic image sanity checks (size / brightness)
    # ----------------------------------------------------
    ok, msg = validate_image_size(image)
    if not ok:
        raise NoFaceDetectedError(msg)
    ok, msg = validate_brightness(image)
    if not ok:
        raise NoFaceDetectedError(msg)

    # ----------------------------------------------------
    # STEP 1: Detect a face in the image
    # This prevents the model from predicting on backgrounds.
    # ----------------------------------------------------
    cropped_face, detection = detect_and_crop_face(image)
    if detection is None:
        raise NoFaceDetectedError()

    # Validate the cropped face region before running inference
    if not _validate_cropped_face(cropped_face):
        raise NoFaceDetectedError(
            "The detected region does not appear to be a valid face. "
            "Please ensure your face is clearly visible with good lighting."
        )

    # ----------------------------------------------------
    # STEP 2: Run ViT on the cropped face
    # ----------------------------------------------------
    return _run_vit_on_face(cropped_face)


def predict_image_with_face_check(image: Image.Image, mode: str) -> Tuple[str, str, float]:
    """
    Unified image prediction entrypoint for all image sources.

    This function enforces the face-detection-first rule for both real
    and mock modes, so webcam frames, uploads, and demo mode all share
    the exact same validation path.

    The pipeline:
    1. Detect face(s) and crop the largest face region
    2. Validate the cropped region contains an actual face (variance/size check)
    3. Run inference only if validation passes

    Args:
        image: PIL Image in RGB format.
        mode: "real" for ViT inference, "mock" for demo predictions.

    Returns:
        emotion, stress_level, confidence

    Raises:
        NoFaceDetectedError when no face is found or the detection is invalid.
    """
    cropped_face, detection = detect_and_crop_face(image)
    if detection is None:
        raise NoFaceDetectedError()

    # Validate the cropped face region before running inference
    # This prevents the model from processing false positives
    if not _validate_cropped_face(cropped_face):
        raise NoFaceDetectedError(
            "The detected region does not appear to be a valid face. "
            "Please ensure your face is clearly visible with good lighting."
        )

    # Note: We now automatically select the largest face from detectors,
    # so we do not error on multiple faces. The detection result already
    # contains the largest face from the chosen detector.

    if mode == "real":
        return _run_vit_on_face(cropped_face)

    # Even in mock mode we still require a visible face to keep UX consistent.
    return predict_mock_from_image()


def predict_from_manual_input(mood: str, stress_scale: int) -> Tuple[str, str, float]:
    """
    Analyze manual self-reported mood and stress input.
    
    Simple rule-based analysis for user self-assessment.
    
    Args:
        mood: User's reported mood/emotion
        stress_scale: User's reported stress level (1-10)
    
    Returns:
        Tuple of (emotion, stress_level, confidence)
    """
    # Calculate stress level from scale
    if stress_scale < 4:
        stress_level = "low"
    elif stress_scale > 7:
        stress_level = "high"
    else:
        stress_level = "medium"
    
    # Confidence based on self-report reliability
    confidence = 0.85
    
    return mood, stress_level, confidence


def predict_from_sensor_data(
    heart_rate: float,
    hrv: float,
    sleep_hours: float,
    stress_scale: int
) -> Tuple[str, str, float]:
    """
    Analyze physiological sensor data to determine stress level.
    
    This function implements a multi-factor stress analysis:
    1. Heart Rate Analysis: Elevated HR indicates stress
    2. HRV Analysis: Lower HRV indicates higher stress
    3. Sleep Analysis: Poor sleep increases stress
    4. Self-Report: User's perceived stress
    
    Args:
        heart_rate: Heart rate in BPM
        hrv: Heart Rate Variability in ms
        sleep_hours: Hours of sleep last night
        stress_scale: User's self-reported stress (1-10)
    
    Returns:
        Tuple of (emotion, stress_level, confidence)
    """
    # -----------------------------------------------------------------------------
    # Factor 1: Heart Rate Analysis
    # -----------------------------------------------------------------------------
    # Normal: 60-100 BPM
    # Elevated HR often indicates stress/arousal
    if heart_rate > 100:
        hr_stress = 3  # High stress indicator
    elif heart_rate > 85:
        hr_stress = 2  # Medium
    else:
        hr_stress = 1  # Low/normal
    
    # -----------------------------------------------------------------------------
    # Factor 2: HRV Analysis
    # -----------------------------------------------------------------------------
    # Higher HRV = better stress regulation
    # Lower HRV = higher stress
    if hrv < 30:
        hrv_stress = 3
    elif hrv < 50:
        hrv_stress = 2
    else:
        hrv_stress = 1
    
    # -----------------------------------------------------------------------------
    # Factor 3: Sleep Analysis
    # -----------------------------------------------------------------------------
    # Optimal: 7-9 hours
    # Less sleep = more stress
    if sleep_hours < 5:
        sleep_stress = 3
    elif sleep_hours > 7:
        sleep_stress = 1
    else:
        sleep_stress = 2
    
    # -----------------------------------------------------------------------------
    # Factor 4: Self-reported stress
    # -----------------------------------------------------------------------------
    # Normalize to 1-3 scale
    self_stress = min(stress_scale / 3.5, 3)
    
    # -----------------------------------------------------------------------------
    # Calculate combined stress score
    # -----------------------------------------------------------------------------
    avg_stress = (hr_stress + hrv_stress + sleep_stress + self_stress) / 4
    
    # Map to categorical stress level
    if avg_stress >= 2.5:
        stress_level = "high"
        confidence = 0.82
    elif avg_stress >= 1.5:
        stress_level = "medium"
        confidence = 0.75
    else:
        stress_level = "low"
        confidence = 0.88
    
    # Generate reason string for transparency
    reason = (
        f"HR={heart_rate}, HRV={hrv}, "
        f"Sleep={sleep_hours}h, Combined={avg_stress:.2f}"
    )
    
    return stress_level, confidence, reason


# =============================================================================
# Mock Inference Functions (for demo/testing)
# =============================================================================

def predict_mock_from_image() -> Tuple[str, str, float]:
    """
    Generate mock prediction for demonstration purposes.
    
    Returns random but realistic emotion/stress combinations.
    Useful for:
    - Frontend development without ML backend
    - Demo Mode when model isn't available
    - Testing API endpoints
    
    Returns:
        Tuple of (emotion, stress_level, confidence)
    """
    emotions = ["happy", "sad", "neutral", "angry", "fear", "disgust", "surprise"]
    emotion = random.choice(emotions)
    confidence = round(random.uniform(0.6, 0.95), 2)
    stress_level = random.choice(["low", "medium", "high"])
    
    return emotion, stress_level, confidence


def predict_mock_from_manual(mood: str, stress_scale: int) -> Tuple[str, str, float]:
    """
    Generate mock prediction for manual input.
    
    Args:
        mood: User's reported mood
        stress_scale: User's reported stress (1-10)
    
    Returns:
        Tuple of (emotion, stress_level, confidence)
    """
    confidence = 0.80
    stress_level = "low" if stress_scale < 4 else "high" if stress_scale > 7 else "medium"
    
    return mood, stress_level, confidence


def predict_mock_from_sensor(
    heart_rate: float,
    stress_scale: int
) -> Tuple[str, str, float]:
    """
    Generate mock prediction for sensor data.
    
    Simple heuristic for demo purposes.
    
    Args:
        heart_rate: Heart rate in BPM
        stress_scale: User's reported stress (1-10)
    
    Returns:
        Tuple of (emotion, stress_level, confidence)
    """
    # Simple mock logic
    if heart_rate > 90 or stress_scale > 7:
        stress_level = "high"
        confidence = 0.82
    elif heart_rate < 70 and stress_scale < 4:
        stress_level = "low"
        confidence = 0.88
    else:
        stress_level = "medium"
        confidence = 0.75
    
    return stress_level, confidence
