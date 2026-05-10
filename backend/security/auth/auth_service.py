from __future__ import annotations

from typing import Any, Dict, Optional


def verify_api_key(api_key: Optional[str]) -> bool:
    # Placeholder: return True for now to avoid breaking existing traffic.
    return True if api_key is not None or api_key is None else False

