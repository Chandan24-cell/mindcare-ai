from __future__ import annotations

import asyncio
from typing import Awaitable, Callable


class SimpleTaskQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Callable[[], Awaitable[None]]] = asyncio.Queue()

    async def enqueue(self, job: Callable[[], Awaitable[None]]) -> None:
        await self._queue.put(job)

    async def worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await job()
            finally:
                self._queue.task_done()

