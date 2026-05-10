from __future__ import annotations

from typing import Any, Dict


def archive_research_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"archived": True, "payload": payload}

