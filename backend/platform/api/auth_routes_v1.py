from __future__ import annotations

from fastapi import APIRouter

from backend.platform.auth.auth_routes import router as auth_router

router = APIRouter()

# Mount legacy scaffolding under /api/v1 automatically from platform router.
router.include_router(auth_router)

