from __future__ import annotations

from fastapi import APIRouter

from backend.infrastructure.health.system_health_service import get_health

router = APIRouter()


@router.get("/health/detailed")
async def health_detailed():
    return get_health(detailed=True)

