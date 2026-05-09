# =============================================================================
# Pydantic Request/Response Schemas
# =============================================================================
# This module defines the data models for API requests and responses.
# Using Pydantic ensures data validation and provides automatic documentation.
# =============================================================================

from pydantic import BaseModel, ConfigDict, Field
from typing import List


# =============================================================================
# Request Models
# =============================================================================

class ManualInput(BaseModel):
    """
    Schema for manual mood/stress self-assessment input.
    """
    mood: str = Field(..., min_length=2, max_length=50)
    stress_scale: int = Field(..., ge=1, le=10)


class SensorInput(BaseModel):
    """
    Schema for physiological sensor data input.
    """
    model_config = ConfigDict(extra="forbid")

    heart_rate: float = Field(..., ge=30, le=220)
    hrv: float = Field(..., ge=1, le=300)
    sleep_hours: float = Field(..., ge=0, le=24)
    activity_level: int = Field(..., ge=1, le=10, description="Physical activity level from 1 (sedentary) to 10 (very active)")
    self_mood: str = Field(..., min_length=2, max_length=50)
    stress_scale: int = Field(..., ge=1, le=10)


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
