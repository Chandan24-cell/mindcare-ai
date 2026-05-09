# =============================================================================
# FastAPI Main Application
# =============================================================================
# This is the main entry point for the Stress Detection API.
# It defines all API endpoints and serves the frontend.
#
# Why This Structure:
# - main.py: FastAPI app, routes, and server configuration
# - model_loader.py: ML model loading and management
# - inference.py: Prediction logic (real and mock)
# - schemas.py: Pydantic models for validation
# - suggestion_engine.py: Wellness recommendations
#
# How Modules Connect:
# 1. Request comes to main.py endpoint
# 2. main.py validates using schemas.py
# 3. main.py calls inference.py for predictions
# 4. inference.py may call model_loader.py for real ML
# 5. main.py calls suggestion_engine.py for recommendations
# 6. Response is returned with proper formatting
#
# Run with: uvicorn backend.main:app --reload
# =============================================================================

import io
import gc
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
from backend.api.request_utils import (
    format_validation_errors,
    payload_dict,
    read_request_payload,
    sensor_activity_label,
)
from backend.schemas import ManualInput, SensorInput
from backend.utils.response import success_response, error_response


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_INFERENCE_LOCK = threading.Lock()
MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024
MAX_DETECTION_IMAGE_SIZE = (640, 640)
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


# =============================================================================
# FastAPI Application Setup
# =============================================================================

# Create FastAPI application instance
app = FastAPI(
    title="Stress & Mental State Detection API",
    description="""
    AI-Powered Mental Wellness Platform
    
    This API provides:
    - Image-based emotion detection using Vision Transformers (ViT)
    - Manual mood/stress self-assessment
    - Physiological sensor data analysis
    - Personalized wellness recommendations
    
    ## Modes
    - **Real Mode**: Uses actual ML model for predictions
    - **Mock Mode**: Returns simulated predictions for demonstration
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# =============================================================================
# Startup Event - Diagnostics
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information and verify configuration."""
    logger.info("%s", "="*70)
    logger.info("MINDCARE BACKEND STARTUP")
    logger.info("%s", "="*70)
    logger.info("API Version: 2.0.0")
    logger.info("Environment: %s", os.getenv('ENVIRONMENT', 'development'))
    logger.info("Port: %s", os.getenv('PORT', '7860'))
    logger.info("CORS enabled for allowed origins only")
    logger.info("ML model loading: lazy, first real image inference only")
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY not set. AI-powered recommendations will be disabled.")
    logger.info("%s", "="*70)
    logger.info("Ready to serve requests.")


# =============================================================================
# CORS Middleware Configuration
# =============================================================================
# Allows cross-origin requests for frontend development
# In production, you might want to restrict this to your frontend domain

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    errors = exc.errors()
    readable_error = format_validation_errors(errors)
    payload = await read_request_payload(request)

    if request.url.path == "/predict/sensor":
        logger.warning("/predict/sensor validation error: %s", readable_error)
        logger.warning("/predict/sensor received payload: %s", payload)
        logger.warning("/predict/sensor validation details: %s", errors)
    else:
        logger.warning(
            "%s %s validation error: %s",
            request.method,
            request.url.path,
            readable_error,
        )

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": readable_error,
            "detail": readable_error,
            "validation_errors": jsonable_encoder(errors),
        },
    )


# =============================================================================
# Root Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Serve the MindCare dashboard."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health():
    """Simple health probe used by deployments and smoke tests."""
    return {"status": "ok"}


@app.get("/favicon.ico")
async def favicon():
    """
    Serve a simple favicon to prevent 404 errors from browser requests.
    Returns a minimal valid ICO file.
    """
    # Return a 1x1 transparent PNG encoded as data URL response
    return RedirectResponse(
        url="data:image/x-icon;base64,AAABAAEAEBAAAAEAIA"
             "BoBAAFgIAAFgIAACAgAAAIAAgAKAgAANgIAAjoAgAAKAAAAEAAAAAB"
             "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
             "AAAAAAAAAA==",
        status_code=200
    )


# =============================================================================
# Image-based Emotion Detection
# =============================================================================

@app.post("/predict/image")
async def predict_from_image(
    file: UploadFile = File(...),
    mode: str = Query("real", description="Prediction mode: 'real' or 'mock'")
):
    """Analyze an uploaded image without changing the prediction contract."""

    if not IMAGE_INFERENCE_LOCK.acquire(blocking=False):
        await file.close()
        return error_response(
            "Image inference is already running. Please retry in a moment.",
            status_code=503,
        )

    try:
        from backend.face_detection import NoFaceDetectedError, MultipleFacesDetectedError
        from backend.inference import predict_image_with_face_check
        from backend.model_loader import (
            MemoryBudgetExceededError,
            ModelUnavailableError,
            ensure_memory_within_limit,
            is_out_of_memory_error,
            log_memory_usage,
            release_unused_memory,
        )
    except Exception as exc:
        IMAGE_INFERENCE_LOCK.release()
        await file.close()
        gc.collect()
        logger.exception("Image prediction dependencies could not be loaded")
        return error_response(
            f"Image prediction dependencies could not be loaded: {exc}",
            status_code=503,
        )

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        IMAGE_INFERENCE_LOCK.release()
        await file.close()
        return error_response(
            "Only JPEG, PNG, and WEBP images are allowed",
            status_code=400,
        )

    contents = None
    image_bytes = None
    image = None
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            return error_response(
                "File too large. Max size is 5MB",
                status_code=413,
            )

        log_memory_usage("predict/image request received")
        ensure_memory_within_limit("image decode", hard_fraction=0.96)

        image_bytes = io.BytesIO(contents)
        contents = None
        with Image.open(image_bytes) as uploaded_image:
            uploaded_image.thumbnail(
                MAX_DETECTION_IMAGE_SIZE,
                Image.Resampling.BILINEAR,
            )
            image = uploaded_image.convert("RGB")

        ensure_memory_within_limit("image preprocessing", hard_fraction=0.97)

        # Run unified face-first prediction pipeline in a threadpool
        emotion, stress_level, confidence = await run_in_threadpool(
            predict_image_with_face_check,
            image,
            mode
        )

        # Build response metadata
        if mode == "real":
            reason = f"Real ML: ViT model detected '{emotion}' with {confidence:.2f} confidence"
            disclaimer = "ML prediction - Approx 70-85% accuracy"
        else:
            reason = f"Mock: Random emotion '{emotion}' for demo"
            disclaimer = "Mock mode - Not real predictions"

        # Get wellness suggestions based on results
        from backend.suggestion_engine import get_suggestions
        suggestions = get_suggestions(emotion, stress_level)

        # Return standardized success response
        return success_response(
            mode=mode,
            emotion=emotion,
            stress_level=stress_level,
            confidence=confidence,
            suggestion=suggestions,
            message=reason,
        )
    except (NoFaceDetectedError, MultipleFacesDetectedError) as e:
        return error_response(str(e), status_code=400)
    except ModelUnavailableError as e:
        return error_response(str(e), status_code=503)
    except MemoryBudgetExceededError as e:
        logger.warning("Image prediction rejected due to memory pressure: %s", e)
        return error_response(str(e), status_code=503)
    except MemoryError as e:
        logger.exception("Image prediction failed due to memory exhaustion")
        return error_response(
            "Image prediction could not complete because the server is low on memory. Please retry.",
            status_code=503,
        )
    except RuntimeError as e:
        if is_out_of_memory_error(e):
            logger.exception("Image prediction failed due to low memory")
            return error_response(
                "Image prediction could not complete because the server is low on memory. Please retry.",
                status_code=503,
            )
        logger.exception("Image prediction runtime error")
        return error_response(f"Image processing error: {str(e)}", status_code=500)
    except Exception as e:
        logger.exception("Image prediction failed")
        return error_response(f"Image processing error: {str(e)}", status_code=500)
    finally:
        if image is not None:
            image.close()
        if image_bytes is not None:
            image_bytes.close()
        await file.close()
        contents = None
        image_bytes = None
        IMAGE_INFERENCE_LOCK.release()
        release_unused_memory()
        log_memory_usage("predict/image cleanup")


# =============================================================================
# Manual Self-Assessment
# =============================================================================

@app.post("/predict/manual")
async def predict_from_manual(
    input: ManualInput,
    mode: str = Query("real", description="Prediction mode: 'real' or 'mock'")
):
    """
    Analyze manual self-reported mood and stress data.
    
    This endpoint accepts user-reported mood and stress scale values
    and provides analysis based on the self-assessment.
    
    ## Parameters:
    - **input**: ManualInput object containing:
        - mood: User's emotional state (happy, sad, angry, etc.)
        - stress_scale: Self-reported stress (1-10)
    - **mode**: Either "real" or "mock"
    
    ## Returns:
    - Reported emotion
    - Calculated stress level
    - Confidence score
    - Analysis explanation
    - Personalized suggestions
    """
    from backend.inference import predict_from_manual_input, predict_mock_from_manual

    mood = input.mood
    stress_scale = input.stress_scale

    # Delegate to inference helpers (real or mock)
    emotion, stress_level, confidence = (
        predict_from_manual_input(mood, stress_scale)
        if mode == "real" else
        predict_mock_from_manual(mood, stress_scale)
    )

    reason = f"Manual input analyzed: mood={mood}, stress scale={stress_scale}/10"
    from backend.suggestion_engine import get_suggestions
    suggestions = get_suggestions(emotion, stress_level)

    return success_response(
        mode=mode,
        emotion=emotion,
        stress_level=stress_level,
        confidence=confidence,
        suggestion=suggestions,
        message=reason,
    )


# =============================================================================
# Sensor Data Analysis
# =============================================================================

@app.post("/predict/sensor")
async def predict_from_sensor(
    request: Request,
    input: SensorInput,
    mode: str = Query("real", description="Prediction mode: 'real' or 'mock'")
):
    """
    Analyze physiological sensor data for stress detection.
    
    This endpoint accepts biometric sensor readings and uses
    multi-factor analysis to determine stress levels.
    
    ## Parameters:
    - **input**: SensorInput object containing:
        - heart_rate: Heart rate in BPM
        - hrv: Heart Rate Variability
        - sleep_hours: Hours of sleep
        - activity_level: Physical activity level from 1 (sedentary) to 10 (very active)
        - self_mood: User's reported mood
        - stress_scale: User's stress self-assessment
    - **mode**: Either "real" or "mock"
    
    ## Analysis Factors:
    1. Heart Rate - Elevated HR indicates stress
    2. HRV - Lower HRV = higher stress
    3. Sleep - Poor sleep increases stress
    4. Self-report - User's perceived stress
    
    ## Returns:
    - Primary emotion (from self-report)
    - Calculated stress level
    - Confidence score
    - Detailed analysis
    - Wellness recommendations
    """
    try:
        if mode not in {"real", "mock"}:
            return error_response(
                "Invalid mode value: expected 'real' or 'mock'",
                status_code=400,
            )

        received_payload = await read_request_payload(request)
        parsed_payload = payload_dict(input)
        logger.info("/predict/sensor received payload: %s", received_payload)
        logger.info("/predict/sensor parsed payload: %s", parsed_payload)

        from backend.inference import predict_from_sensor_data, predict_mock_from_sensor

        # Extract sensor values
        heart_rate = input.heart_rate
        hrv = input.hrv
        sleep_hours = input.sleep_hours
        self_mood = input.self_mood
        stress_scale = input.stress_scale
        activity_level = sensor_activity_label(input.activity_level)
        logger.info(
            "/predict/sensor normalized activity_level: %s -> %s",
            input.activity_level,
            activity_level,
        )

        if mode == "real":
            stress_level, confidence, sensor_reason = predict_from_sensor_data(
                heart_rate=heart_rate,
                hrv=hrv,
                sleep_hours=sleep_hours,
                stress_scale=stress_scale,
                activity_level=activity_level
            )
            reason = f"Real ML: {sensor_reason}"
        else:
            stress_level, confidence = predict_mock_from_sensor(
                heart_rate=heart_rate,
                stress_scale=stress_scale
            )[:2]
            reason = (
                f"Mock: HR {heart_rate}, stress scale {stress_scale}, "
                f"activity {activity_level}"
            )

        # Get suggestions based on results
        from backend.suggestion_engine import get_suggestions
        suggestions = get_suggestions(self_mood, stress_level)

        logger.info(
            "/predict/sensor response: emotion=%s stress_level=%s confidence=%s",
            self_mood,
            stress_level,
            confidence,
        )
        return success_response(
            mode=mode,
            emotion=self_mood,
            stress_level=stress_level,
            confidence=confidence,
            suggestion=suggestions,
            message=reason,
        )
    except Exception as exc:
        logger.exception("/predict/sensor processing failed")
        return error_response(
            f"Sensor processing error: {exc}",
            status_code=500,
        )


# =============================================================================
# Report Generation
# =============================================================================

@app.post("/generate-report")
async def generate_pdf_report(request: dict):
    """
    Generate a PDF report of the analysis results.

    Accepts JSON payload with analysis data and returns the path to generated PDF.

    Request body:
        email: User's email address
        emotion: Detected emotion
        stress_level: Calculated stress level (low/medium/high)
        confidence: Model confidence score (0-1)
        suggestions: List of wellness recommendations
    """
    try:
        from backend.report_generator import generate_report

        filepath = generate_report(
            email=request.get("email", "user@example.com"),
            emotion=request.get("emotion", "Unknown"),
            stress_level=request.get("stress_level", "Unknown"),
            confidence=request.get("confidence", 0.0),
            suggestions=request.get("suggestions") or request.get("suggestion", []),
            reason=request.get("reason"),
        )
        return {"status": "success", "report_path": filepath}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


# =============================================================================
# Static File Serving
# =============================================================================

# Mount the frontend directory to serve static files (HTML aware)
# This allows the API to serve the web application directly
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# Mount reports directory for PDF downloads
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


# =============================================================================
# Server Startup
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    
    print("\n" + "="*60)
    print("Starting Stress Detection API Server v2.0")
    print("="*60)
    print("Backend Structure:")
    print("   main.py              (routes and static serving)")
    print("   model_loader.py      (ViT model management)")
    print("   inference.py         (prediction logic)")
    print("   schemas.py           (data validation)")
    print("   suggestion_engine.py (recommendations)")
    print("="*60)
    print("Default Mode: Real ML predictions")
    print("Real ML Mode: ViT emotion detection (lazy loaded)")
    print(f"Server: http://localhost:{port}")
    print(f"API Docs: http://localhost:{port}/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
