from __future__ import annotations

from typing import Any, Awaitable, Callable


class BackgroundTaskQueue:
    """No-op background queue placeholder."""

    def __init__(self) -> None:
        pass

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        # execute synchronously for determinism in current phase
        fn(*args, **kwargs)

