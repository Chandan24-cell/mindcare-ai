from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Counter:
    value: int = 0

    def inc(self, n: int = 1) -> None:
        self.value += int(n)


class MetricsRegistry:
    """In-process metrics registry.

    This is a lightweight placeholder to prepare for Prometheus/Grafana.
    """

    def __init__(self) -> None:
        self.counters: Dict[str, Counter] = {}

    def counter(self, name: str) -> Counter:
        if name not in self.counters:
            self.counters[name] = Counter(0)
        return self.counters[name]

    def snapshot(self) -> Dict[str, int]:
        return {k: v.value for k, v in self.counters.items()}


registry = MetricsRegistry()

