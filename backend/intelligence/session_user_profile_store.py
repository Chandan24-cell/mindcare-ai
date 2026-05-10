from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class UserProfileStore:
    def __init__(
        self,
        *,
        storage_path: Path,
        max_users: int = 500,
    ) -> None:
        self._storage_path = storage_path
        self._lock = threading.Lock()
        self._max_users = max_users
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _read(self) -> Dict[str, Any]:
        try:
            if not self._storage_path.exists():
                return {"users": {}}
            raw = self._storage_path.read_text(encoding="utf-8")
            if not raw.strip():
                return {"users": {}}
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(data.get("users"), dict):
                return data
            return {"users": {}}
        except Exception:
            logger.warning("User profiles JSON corrupted/unreadable; using empty store", exc_info=True)
            return {"users": {}}

    def _write(self, data: Dict[str, Any]) -> None:
        tmp = self._storage_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self._storage_path)
        except Exception:
            logger.warning("Failed to write user profiles JSON", exc_info=True)

    def get_user(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            data = self._read()
            users = data.get("users", {})
            user = users.get(user_id)
            return user if isinstance(user, dict) else {}

    def upsert_user(self, user_id: str, patch: Dict[str, Any]) -> None:
        with self._lock:
            data = self._read()
            users = data.setdefault("users", {})
            cur = users.get(user_id, {})
            if not isinstance(cur, dict):
                cur = {}
            # Shallow merge
            cur.update(patch)
            users[user_id] = cur

            # bounded storage by dropping oldest keys is complex; enforce rough bound.
            if len(users) > self._max_users:
                # keep last N arbitrary ordering
                items = list(users.items())[-self._max_users:]
                users = {k: v for k, v in items}
                data["users"] = users

            self._write(data)

    def get_or_create_user(self, user_id: str, default: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            data = self._read()
            users = data.setdefault("users", {})
            user = users.get(user_id)
            if not isinstance(user, dict) or not user:
                users[user_id] = default
                self._write(data)
                return default
            return user


def default_user_profile_store() -> UserProfileStore:
    project_root = Path(__file__).resolve().parents[1]
    storage_path = project_root / "data" / "user_profiles.json"
    return UserProfileStore(storage_path=storage_path)

