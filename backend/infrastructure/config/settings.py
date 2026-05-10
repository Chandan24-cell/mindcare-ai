from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class AppSettings:
    """Lightweight settings container.

    This intentionally does NOT introduce hard dependencies on pydantic
    settings to avoid runtime breakage.
    """

    environment: str
    port: int
    host: str

    # Observability toggles
    enable_metrics: bool

    # Security placeholders
    enable_rate_limit: bool
    api_key_required: bool


def load_settings() -> AppSettings:
    env = os.getenv("ENVIRONMENT", "development").strip().lower()
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "127.0.0.1").strip()

    enable_metrics = os.getenv("ENABLE_METRICS", "true").strip().lower() in {"1", "true", "yes"}
    enable_rate_limit = os.getenv("ENABLE_RATE_LIMIT", "false").strip().lower() in {"1", "true", "yes"}
    api_key_required = os.getenv("API_KEY_REQUIRED", "false").strip().lower() in {"1", "true", "yes"}

    return AppSettings(
        environment=env,
        port=port,
        host=host,
        enable_metrics=enable_metrics,
        enable_rate_limit=enable_rate_limit,
        api_key_required=api_key_required,
    )

