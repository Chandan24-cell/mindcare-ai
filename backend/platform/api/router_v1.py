from __future__ import annotations

from fastapi import APIRouter

from backend.platform.api.auth_routes_v1 import router as auth_routes
from backend.platform.api.users_routes_v1 import router as users_routes

router = APIRouter(prefix="/api/v1")
router.include_router(auth_routes)
router.include_router(users_routes)

