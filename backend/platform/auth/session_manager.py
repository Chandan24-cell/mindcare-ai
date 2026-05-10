from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SessionState:
    access_token: str
    user_id: str


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def create(self, *, access_token: str, user_id: str) -> None:
        async with self._lock:
            self._sessions[access_token] = SessionState(access_token=access_token, user_id=user_id)

    async def exists(self, access_token: str) -> bool:
        async with self._lock:
            return access_token in self._sessions

    async def delete(self, access_token: str) -> None:
        async with self._lock:
            self._sessions.pop(access_token, None)


session_manager = SessionManager()

