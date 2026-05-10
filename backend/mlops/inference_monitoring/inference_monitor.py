from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class InferenceRecord:
    endpoint: str
    mode: str
    success: bool
    latency_ms: float


def record_inference(*, endpoint: str, mode: str, success: bool, latency_ms: float) -> None:
    # Placeholder for future persistent storage.
    return

