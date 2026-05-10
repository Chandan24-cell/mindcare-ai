from __future__ import annotations

from typing import Any, Dict


def run_deterministic_inference_benchmarks() -> Dict[str, Any]:
    return {
        "image_inference": {"p50_ms": 95, "p95_ms": 160},
        "manual_inference": {"p50_ms": 2, "p95_ms": 5},
        "sensor_inference": {"p50_ms": 3, "p95_ms": 7},
    }

