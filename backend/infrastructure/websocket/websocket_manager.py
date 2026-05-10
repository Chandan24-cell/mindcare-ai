from __future__ import annotations

from typing import Any, Dict, Optional


class WebSocketManager:
    """Placeholder websocket manager for future streaming readiness."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Any] = {}

    async def connect(self, session_id: str, websocket: Any) -> None:
        self._sessions[session_id] = websocket

    async def disconnect(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        # No-op placeholder.
        return

