# =============================================================================
# Pydantic Request/Response Schemas
# =============================================================================
# This module defines the data models for API requests and responses.
# Using Pydantic ensures data validation and provides automatic documentation.
# =============================================================================

from pydantic import BaseModel
from typing import Optional, List


# =============================================================================
# Request Models
# =============================================================================

class ManualInput(BaseModel):
    """
    Schema for manual mood/stress self-assessment input.
    
    Attributes:
        mood: User's self-reported emotional state
        stress_scale: User's self-reported stress level (1-10)
    """
    mood: str
    stress_scale: int


class SensorInput(BaseModel):
    """
    Schema for physiological sensor data input.
    
    Attributes:
        heart_rate: Heart rate in beats per minute (bpm)
        hrv: Heart Rate Variability (ms)
        sleep_hours: Hours of sleep
        activity_level: User's activity level (1-10)
        self_mood: User's self-reported mood
        stress_scale: User's self-reported stress level (1-10)
    """
    heart_rate: float
    hrv: float
    sleep_hours: float
    activity_level: float
    self_mood: str
    stress_scale: int


# =============================================================================
# Response Models
# =============================================================================

class PredictionResponse(BaseModel):
    """
    Standard response schema for all prediction endpoints.
    
    Attributes:
        emotion: Detected or reported emotion
        stress_level: Calculated stress level (low/medium/high)
        confidence: Model confidence score (0-1)
        reason: Human-readable explanation of the prediction
        suggestion: List of wellness recommendations
        disclaimer: Mode disclaimer (real/ml/mock)
        mode: Prediction mode (real/mock)
    """
    emotion: str
    stress_level: str
    confidence: float
    reason: str
    suggestion: List[str]
    disclaimer: str
    mode: str

