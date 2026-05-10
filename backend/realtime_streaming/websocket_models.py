from __future__ import annotations

from typing import Any, Dict, List, Optional

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamMetrics:
    stream_fps: float
    avg_latency_ms: float
    dropped_frames: int
    stream_quality: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_fps": self.stream_fps,
            "avg_latency_ms": self.avg_latency_ms,
            "dropped_frames": self.dropped_frames,
            "stream_quality": self.stream_quality,
        }

