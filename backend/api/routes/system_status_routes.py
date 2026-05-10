from __future__ import annotations

from fastapi import APIRouter

from backend.observability.diagnostics.diagnostics_service import get_system_status

router = APIRouter()


@router.get("/system/status")
async def system_status():
    return get_system_status()

