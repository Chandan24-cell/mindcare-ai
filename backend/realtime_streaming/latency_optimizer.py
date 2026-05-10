from __future__ import annotations


def optimize_latency_params() -> dict:
    return {
        "frame_skip": 1,
        "target_latency_ms": 150,
        "max_inference_ms": 120,
    }

