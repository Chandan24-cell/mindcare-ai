from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.analytics.analytics_models import SessionRecord, now_iso

logger = logging.getLogger(__name__)


class SessionHistoryStore:
    """Thread-safe JSON session history store.

    - Stores sessions under backend/data/session_history.json
    - Handles missing/invalid JSON gracefully.
    - Keeps a bounded history size.
    """

    def __init__(
        self,
        *,
        storage_path: Path,
        max_sessions: int = 50,
    ) -> None:
        self._storage_path = storage_path
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

        # Ensure directory exists, but do not fail if filesystem is read-only.
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # We'll still attempt best-effort writes later.
            pass

    def _read_all(self) -> List[Dict[str, Any]]:
        try:
            if not self._storage_path.exists():
                return []
            raw = self._storage_path.read_text(encoding="utf-8")
            if not raw.strip():
                return []
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "sessions" in data and isinstance(
                data["sessions"], list
            ):
                return data["sessions"]
            return []
        except Exception:
            # Corruption handling: don't crash prediction endpoints.
            logger.warning(
                "Session history JSON corrupted/unreadable; falling back to empty store.",
                exc_info=True,
            )
            return []

    def _write_all(self, sessions: List[Dict[str, Any]]) -> None:
        # Best-effort safe write using temp file + atomic replace.
        try:
            tmp_path = self._storage_path.with_suffix(".tmp")
            payload: Dict[str, Any] = {"sessions": sessions}
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, self._storage_path)
        except Exception:
            logger.warning(
                "Failed to write session history; continuing without persistence.",
                exc_info=True,
            )

    def append_session(self, record: SessionRecord) -> List[Dict[str, Any]]:
        """Append a session record and return updated sessions list."""
        with self._lock:
            sessions = self._read_all()
            sessions.append(record.to_dict())

            # Sort newest-first by timestamp if parseable; fallback to insertion order.
            try:
                sessions.sort(
                    key=lambda s: s.get("timestamp", ""), reverse=True
                )
            except Exception:
                pass

            # Keep bounded
            sessions = sessions[-self._max_sessions :]
            self._write_all(sessions)
            return sessions

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            sessions = self._read_all()
            sessions = sessions[-limit:]
            return sessions


def default_store() -> SessionHistoryStore:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "session_history.json"
    return SessionHistoryStore(storage_path=data_path)


def build_session_record(
    *,
    session_id: str,
    emotion: str,
    stress_level: str,
    confidence: float,
    wellness_score: int,
    severity: str,
    source_modalities: List[str],
    sleep_hours: Optional[float] = None,
    heart_rate: Optional[float] = None,
    hrv: Optional[float] = None,
    activity_level: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        timestamp=timestamp or now_iso(),
        emotion=emotion,
        stress_level=stress_level,
        confidence=float(confidence),
        wellness_score=int(wellness_score),
        severity=severity,
        sleep_hours=sleep_hours,
        heart_rate=heart_rate,
        hrv=hrv,
        activity_level=activity_level,
        source_modalities=source_modalities,
    )

