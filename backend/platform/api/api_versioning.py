from __future__ import annotations

from typing import Optional


def get_api_prefix(version: str = "v1") -> str:
    return f"/api/{version}".rstrip("/")


def versioned_path(path: str, *, version: str = "v1") -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"{get_api_prefix(version)}{path}".replace("//", "/")

