from .health_routes import router as health_router
from .prediction_routes import router as prediction_router
from .report_routes import router as report_router

__all__ = ["health_router", "prediction_router", "report_router"]

