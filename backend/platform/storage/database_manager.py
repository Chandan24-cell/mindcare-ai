from __future__ import annotations

from typing import Any, Optional


class DatabaseManager:
    """Future-ready DB manager placeholder.

    Currently deterministic/no-op to keep additive safety.
    """

    def __init__(self) -> None:
        self._initialized = False

    async def init(self) -> None:
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

