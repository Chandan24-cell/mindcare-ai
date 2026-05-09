"""
Response helpers to standardize API outputs.

Frontend popups rely on consistent shapes, so every endpoint should
use these helpers for success and error payloads.
"""
from typing import List
from fastapi.responses import JSONResponse


def success_response(
    mode: str,
    emotion: str,
    stress_level: str,
    confidence: float,
    suggestion: List[str],
    message: str
) -> JSONResponse:
    """Return the established prediction response contract.

    The frontend consumes ``suggestion`` while API clients may read
    ``suggestions``. Both keys are intentionally preserved.
    """
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "mode": mode,
            "emotion": emotion,
            "stress_level": stress_level,
            "confidence": confidence,
            "suggestion": suggestion,
            "suggestions": suggestion,
            "message": message,
        },
    )


def error_response(message: str, status_code: int = 400) -> JSONResponse:
    """Return the established API error shape."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": message,
        },
    )
