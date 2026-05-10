from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List


class ConversationMemory:
    """In-memory conversation memory placeholder.

    This is designed for future persistence/websocket use.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: List[Dict[str, Any]] = []

    def add(self, message: Dict[str, Any]) -> None:
        if not isinstance(message, dict):
            return
        with self._lock:
            self._messages.append(message)
            self._messages = self._messages[-200:]

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._messages)

