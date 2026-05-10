from __future__ import annotations


def paginate(limit: int = 50, offset: int = 0) -> dict:
    return {"limit": max(1, int(limit)), "offset": max(0, int(offset))}

