from __future__ import annotations

from fastapi import APIRouter

from backend.observability.metrics.metrics_endpoints import get_metrics_payload

router = APIRouter()


@router.get("/metrics")
async def metrics():
    return get_metrics_payload()

