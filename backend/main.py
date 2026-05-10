import gc
import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from PIL import Image

from backend.api.routes.health_routes import router as health_router
from backend.api.routes.prediction_routes import router as prediction_router
from backend.api.routes.report_routes import router as report_router
from backend.api.routes.enterprise_routes import router as enterprise_router
from backend.api.request_utils import format_validation_errors, read_request_payload

from backend.realtime_streaming.websocket_server import router as realtime_router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Stress & Mental State Detection API",
    description="""AI-Powered Mental Wellness Platform""",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

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
    exc: RequestValidationError,
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

    return {
        "success": False,
        "error": readable_error,
        "detail": readable_error,
        "validation_errors": jsonable_encoder(errors),
    }, 422


@app.on_event("startup")
async def startup_event():
    logger.info("MINDCARE BACKEND STARTUP")
    logger.info("Environment: %s", os.getenv("ENVIRONMENT", "development"))


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(
        url="data:image/x-icon;base64,AAABAAEAEBAAAAEAIA"
            "BoBAAFgIAAFgIAACAgAAAIAAgAKAgAANgIAAjoAgAAKAAAAEAAAAAB"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAA==",
        status_code=200,
    )


# include routers
app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(report_router)
app.include_router(enterprise_router)

# platform versioned (additive-only)
from backend.platform.api.router_v1 import router as platform_router_v1
app.include_router(platform_router_v1)

# research endpoints (additive-only)
from backend.research.research_routes import router as research_router
app.include_router(research_router)

# realtime streaming websocket
app.include_router(realtime_router)


@app.on_event("startup")
def _log_registered_routes() -> None:
    # Helps validate router integration at runtime.
    registered = []
    for r in app.routes:
        p = getattr(r, "path", None)
        if p:
            registered.append(p)
    logger.info("REGISTERED ROUTES: %s", sorted(set(registered)))




# Static file serving (unchanged behavior)
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    environment = os.getenv("ENVIRONMENT", "development").strip().lower()
    host = os.getenv(
        "HOST",
        "127.0.0.1" if environment == "development" else "0.0.0.0",
    ).strip()

    uvicorn.run(app, host=host, port=port)

