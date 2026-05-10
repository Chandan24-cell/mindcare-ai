from __future__ import annotations

"""Enterprise routes aggregator.

Kept separate to avoid changing backend/main.py route registration logic
more than needed.
"""

from fastapi import APIRouter

from backend.api.routes.health_detailed_routes import router as health_detailed_router
from backend.api.routes.metrics_routes import router as metrics_router
from backend.api.routes.system_status_routes import router as system_status_router

router = APIRouter()
router.include_router(health_detailed_router)
router.include_router(metrics_router)
router.include_router(system_status_router)

