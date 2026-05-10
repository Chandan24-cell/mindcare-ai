from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException

from backend.platform.api.response_models import Envelope
from backend.platform.auth.auth_models import LoginRequest, LogoutRequest, RegisterRequest, RefreshRequest
from backend.platform.auth.jwt_manager import create_tokens, decode_access_token, decode_refresh_token
from backend.platform.auth.password_manager import hash_password, verify_password
from backend.platform.auth.role_manager import default_roles

from typing import Dict

router = APIRouter()

# Deterministic local user store (compat placeholder)
# NOTE: future phases will persist to SQLite/Postgres.
_USER_STORE: Dict[str, Dict[str, str]] = {}

_SECRET = os.getenv("MINDCARE_PLATFORM_SECRET", "local-dev-secret")


@router.post("/auth/register")
async def register(req: RegisterRequest):
    user_id = req.user_id.strip()
    if user_id in _USER_STORE:
        raise HTTPException(status_code=409, detail="User already exists")

    salt = f"salt::{user_id}"
    pw_hash = hash_password(req.password, salt=salt)
    _USER_STORE[user_id] = {"salt": salt, "pw_hash": pw_hash, "roles": ",".join(default_roles())}

    bundle = create_tokens(user_id=user_id, roles=default_roles(), secret=_SECRET)
    return Envelope(success=True, data={"tokens": {"access_token": bundle.access_token, "refresh_token": bundle.refresh_token, "expires_at": bundle.expires_at}})


@router.post("/auth/login")
async def login(req: LoginRequest):
    user_id = req.user_id.strip()
    rec = _USER_STORE.get(user_id)
    if not rec:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    salt = rec["salt"]
    expected = rec["pw_hash"]
    if not verify_password(req.password, salt=salt, expected_hash=expected):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    roles = rec.get("roles", "user").split(",")
    bundle = create_tokens(user_id=user_id, roles=roles, secret=_SECRET)
    return Envelope(success=True, data={"tokens": {"access_token": bundle.access_token, "refresh_token": bundle.refresh_token, "expires_at": bundle.expires_at}})


@router.post("/auth/refresh")
async def refresh(req: RefreshRequest):
    body = decode_refresh_token(req.refresh_token, secret=_SECRET)
    if not body:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user_id = str(body.get("sub"))
    roles = body.get("roles") or ["user"]
    bundle = create_tokens(user_id=user_id, roles=list(roles), secret=_SECRET)
    return Envelope(success=True, data={"tokens": {"access_token": bundle.access_token, "refresh_token": bundle.refresh_token, "expires_at": bundle.expires_at}})


@router.post("/auth/logout")
async def logout(req: LogoutRequest):
    # Deterministic placeholder: accept and return success.
    body = decode_access_token(req.access_token, secret=_SECRET)
    if not body:
        # For additive safety, still return success to avoid breaking clients.
        return Envelope(success=True, data={"logged_out": False})
    return Envelope(success=True, data={"logged_out": True})

