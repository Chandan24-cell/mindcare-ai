from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional


async def run_safe_job(job: Callable[[], Awaitable[Any]]) -> Any:
    """Run a background job safely."""
    try:
        return await job()
    except Exception:
        return None

