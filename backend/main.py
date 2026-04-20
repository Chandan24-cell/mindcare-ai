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
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import our custom modules
from backend.schemas import ManualInput, SensorInput
from backend.inference import (
    predict_emotion_from_image,
    predict_mock_from_image,
    predict_image_with_face_check,
    predict_from_manual_input,
    predict_mock_from_manual,
    predict_from_sensor_data,
    predict_mock_from_sensor
)
from backend.suggestion_engine import get_suggestions
from backend.face_detection import NoFaceDetectedError, MultipleFacesDetectedError
from backend.model_loader import ModelUnavailableError, get_model_status
from backend.utils.response import success_response, error_response
from backend.report_generator import generate_report


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


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
    import os
    print("\n" + "="*70)
    print("🚀 MINDCARE BACKEND STARTUP")
    print("="*70)
    print(f"✅ API Version: 2.0.0")
    print(f"✅ Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"✅ Port: {os.getenv('PORT', '8000')}")
    print(f"✅ CORS enabled for all origins")
    model_status = get_model_status()
    if model_status["available"]:
        print(f"✅ ML Model: READY ({model_status['path']})")
    else:
        print(f"⚠️  ML Model: NOT AVAILABLE - Running in Mock Mode")
        print(f"   Reason: {model_status['message']}")
    print("="*70)
    print("🌐 Ready to serve requests!\n")


# =============================================================================
# CORS Middleware Configuration
# =============================================================================
# Allows cross-origin requests for frontend development
# In production, you might want to restrict this to your frontend domain

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Root Endpoints
# =============================================================================

@app.get("/")
async def serve_frontend():
    """
    Serve the login page at the root URL.
    
    This is the entry point for the web application.
    Users will be redirected here and then to the main app after authentication.
    """
    return RedirectResponse(url="/frontend/login.html")


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        JSON status message indicating the API is running
    """
    model_status = get_model_status()
    health_message = "Service is healthy"
    if not model_status["available"]:
        health_message = (
            "Service is healthy, but real image mode is unavailable until the ViT "
            "checkpoint is added."
        )

    return success_response(
        mode="health",
        emotion="",
        stress_level="",
        confidence=1.0,
        suggestion=[],
        message=health_message,
    )


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
    """
    Predict emotion and stress level from an uploaded image.
    
    This endpoint accepts an image file (typically a face photo) and uses
    the Vision Transformer (ViT) model to detect emotional state.
    
    ## Parameters:
    - **file**: Image file (JPG, PNG, etc.) containing a face
    - **mode**: Either "real" for ML predictions or "mock" for demo
    
    ## Returns:
    - Detected emotion
    - Inferred stress level (low/medium/high)
    - Model confidence score
    - Explanation of the prediction
    - Wellness recommendations
    
    ## Example:
    ```bash
    curl -X POST "http://localhost:8000/predict/image?mode=real" \\
         -F "file=@face.jpg"
    ```
    """
    try:
        # Read and open the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run unified face-first prediction pipeline (real or mock)
        emotion, stress_level, confidence = predict_image_with_face_check(image, mode)

        # Build response metadata
        if mode == "real":
            reason = f"Real ML: ViT model detected '{emotion}' with {confidence:.2f} confidence"
            disclaimer = "ML prediction - Approx 70-85% accuracy"
        else:
            reason = f"Mock: Random emotion '{emotion}' for demo"
            disclaimer = "Mock mode - Not real predictions"
        
        # Get wellness suggestions based on results
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
    except Exception as e:
        # Handle any errors during processing
        raise HTTPException(
            status_code=500,
            detail=f"Image processing error: {str(e)}"
        )


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
    mood = input.mood
    stress_scale = input.stress_scale

    # Delegate to inference helpers (real or mock)
    emotion, stress_level, confidence = (
        predict_from_manual_input(mood, stress_scale)
        if mode == "real" else
        predict_mock_from_manual(mood, stress_scale)
    )

    reason = f"Manual input analyzed: mood={mood}, stress scale={stress_scale}/10"
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
        - activity_level: Physical activity level
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
    # Extract sensor values
    heart_rate = input.heart_rate
    hrv = input.hrv
    sleep_hours = input.sleep_hours
    self_mood = input.self_mood
    stress_scale = input.stress_scale
    
    if mode == "real":
        stress_level, confidence, sensor_reason = predict_from_sensor_data(
            heart_rate=heart_rate,
            hrv=hrv,
            sleep_hours=sleep_hours,
            stress_scale=stress_scale
        )
        reason = f"Real ML: {sensor_reason}"
    else:
        stress_level, confidence = predict_mock_from_sensor(
            heart_rate=heart_rate,
            stress_scale=stress_scale
        )[:2]
        reason = f"Mock: HR {heart_rate}, stress scale {stress_scale}"
    
    # Get suggestions based on results
    suggestions = get_suggestions(self_mood, stress_level)
    
    return success_response(
        mode=mode,
        emotion=self_mood,
        stress_level=stress_level,
        confidence=confidence,
        suggestion=suggestions,
        message=reason,
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
    
    print("\n" + "="*60)
    print("🚀 Starting Stress Detection API Server v2.0")
    print("="*60)
    print("📁 Backend Structure:")
    print("   ├── main.py              (This file)")
    print("   ├── model_loader.py      (ViT model management)")
    print("   ├── inference.py         (Prediction logic)")
    print("   ├── schemas.py          (Data validation)")
    print("   └── suggestion_engine.py (Recommendations)")
    print("="*60)
    print("✅ Default Mode: Real ML predictions")
    print("🧠 Real ML Mode: ViT emotion detection (lazy loaded)")
    print("🌐 Server: http://localhost:8000")
    print("📊 API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
