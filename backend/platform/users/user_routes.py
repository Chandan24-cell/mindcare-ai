from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.platform.api.response_models import Envelope
from backend.platform.users.user_models import UserProfile

router = APIRouter()

# Deterministic local profiles store.
_USER_PROFILES: dict[str, dict] = {}


@router.get("/users/{user_id}")
async def get_user_profile(user_id: str):
    rec = _USER_PROFILES.get(user_id)
    if not rec:
        # Additive safety: return empty profile rather than 404.
        return Envelope(success=True, data={"profile": UserProfile(user_id=user_id).model_dump()})
    return Envelope(success=True, data={"profile": rec})

