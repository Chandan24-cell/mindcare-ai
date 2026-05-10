from __future__ import annotations

from typing import Any, Dict, List


class StreamProcessor:
    """Async-ready placeholder for future websocket/stream ingestion."""

    def __init__(self) -> None:
        self._buffer: List[Dict[str, Any]] = []

    def push(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        self._buffer.append(event)
        self._buffer = self._buffer[-500:]

    def snapshot(self) -> List[Dict[str, Any]]:
        return list(self._buffer)

