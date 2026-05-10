from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AuditEvent:
    timestamp: str
    event_type: str
    details: Dict[str, Any]


class InMemoryAuditLog:
    def __init__(self) -> None:
        self._events: List[AuditEvent] = []

    def record(self, event_type: str, details: Dict[str, Any] | None = None) -> None:
        details = details or {}
        self._events.append(
            AuditEvent(
                timestamp=datetime.utcnow().isoformat() + "Z",
                event_type=event_type,
                details=details,
            )
        )
        self._events = self._events[-500:]

    def snapshot(self) -> List[Dict[str, Any]]:
        return [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "details": e.details,
            }
            for e in self._events
        ]


audit_log = InMemoryAuditLog()

