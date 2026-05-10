from __future__ import annotations

from typing import Any, Dict


def basic_health() -> Dict[str, Any]:
    return {"status": "ok"}


def detailed_health() -> Dict[str, Any]:
    # Keep deterministic and safe. Avoid importing heavy ML modules.
    return {
        "status": "ok",
        "dependencies": {
            "ml_model": "unknown",
            "email": "unknown",
        },
    }

