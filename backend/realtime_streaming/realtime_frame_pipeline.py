from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional


async def run_realtime_frame_pipeline(
    *,
    frame: Optional[Any],
    mode: str = "mock",
) -> Dict[str, Any]:
    """Placeholder realtime pipeline.

    Phase 8 full inference batching can be added later.
    For now, return deterministic mock payload without breaking websocket.
    """

    # Keep async-safe and low CPU.
    await asyncio.sleep(0)

    emotion = "neutral" if mode == "mock" else "neutral"
    stress_level = "medium"
    confidence = 0.7

    return {
        "prediction": {
            "emotion": emotion,
            "stress_level": stress_level,
            "confidence": confidence,
            "reason": "Realtime placeholder prediction",
            "suggestion": [],
            # additive fields reserved for future
        },
        "affective_analysis": {},
        "facial_signals": {},
        "stream_tracking": {},
        "timeline_analysis": {},
        "cognitive_stream": {},
        "alerts": [],
        "stream_metrics": {
            "stream_fps": 0,
            "avg_latency_ms": 0,
            "dropped_frames": 0,
            "stream_quality": "init",
        },
        "copilot_state": {},
        "timestamp": None,
        "session_id": None,
    }

