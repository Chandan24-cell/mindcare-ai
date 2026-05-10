from __future__ import annotations

from fastapi import APIRouter

from backend.platform.users.user_routes import router as users_router

router = APIRouter()
router.include_router(users_router)

