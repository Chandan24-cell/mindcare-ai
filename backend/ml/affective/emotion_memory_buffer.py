from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class EmotionMemoryBufferItem:
    emotion: str
    confidence: float


class EmotionMemoryBuffer:
    """Rolling buffer for affective temporal stabilization."""

    def __init__(self, maxlen: int = 5):
        self._buf: Deque[EmotionMemoryBufferItem] = deque(maxlen=maxlen)

    def append(self, *, emotion: str, confidence: float) -> None:
        if not emotion:
            return
        try:
            conf = float(confidence)
        except Exception:
            conf = 0.0
        self._buf.append(EmotionMemoryBufferItem(emotion=emotion, confidence=conf))

    def items(self) -> list[EmotionMemoryBufferItem]:
        return list(self._buf)

    def is_empty(self) -> bool:
        return len(self._buf) == 0

    def last(self) -> Optional[EmotionMemoryBufferItem]:
        if not self._buf:
            return None
        return self._buf[-1]

