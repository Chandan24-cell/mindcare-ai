from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Dict, Optional


class Envelope(BaseModel):
    """Standard envelope for future versioned APIs.

    Kept minimal + additive; legacy endpoints remain unchanged.
    """

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

