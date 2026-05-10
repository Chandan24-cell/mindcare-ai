from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    session_id: str
    created_at: float


class StreamSessionManager:
    """In-memory session manager for websocket sessions.

    Defensive + bounded: no unbounded history stored.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def create(self, session_id: str, *, created_at: float) -> SessionState:
        async with self._lock:
            state = SessionState(session_id=session_id, created_at=created_at)
            self._sessions[session_id] = state
            return state

    async def get(self, session_id: str) -> Optional[SessionState]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

