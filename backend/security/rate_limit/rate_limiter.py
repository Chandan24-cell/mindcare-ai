from __future__ import annotations

from typing import Any, Dict


class RateLimiter:
    """No-op rate limiter placeholder."""

    def is_allowed(self, *, key: str) -> bool:
        return True

