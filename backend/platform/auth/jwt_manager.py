from __future__ import annotations

import base64
import hmac
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


# Lightweight deterministic token placeholder.
# Future phases will replace with real JWT + refresh flows.


@dataclass(frozen=True)
class TokenBundle:
    access_token: str
    refresh_token: str
    expires_at: int


def _sign(payload: str, secret: str) -> str:
    sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256)
    return sig.hexdigest()


def create_tokens(*, user_id: str, roles: list[str], secret: str, ttl_seconds: int = 3600) -> TokenBundle:
    now = int(time.time())
    expires_at = now + int(ttl_seconds)

    body_access = {"sub": user_id, "roles": roles, "exp": expires_at, "type": "access"}
    body_refresh = {"sub": user_id, "roles": roles, "exp": expires_at + 7 * 24 * 3600, "type": "refresh"}

    access_payload = base64.urlsafe_b64encode(json.dumps(body_access, separators=(",", ":")).encode("utf-8")).decode("utf-8")
    refresh_payload = base64.urlsafe_b64encode(json.dumps(body_refresh, separators=(",", ":")).encode("utf-8")).decode("utf-8")

    access_token = f"{access_payload}.{_sign(access_payload, secret)}"
    refresh_token = f"{refresh_payload}.{_sign(refresh_payload, secret)}"

    return TokenBundle(access_token=access_token, refresh_token=refresh_token, expires_at=expires_at)


def _verify_token(token: str, *, secret: str) -> Optional[Dict[str, Any]]:
    try:
        payload_b64, sig = token.split(".", 1)
    except ValueError:
        return None

    expected = _sign(payload_b64, secret)
    if not hmac.compare_digest(expected, sig):
        return None

    try:
        payload_raw = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        body = json.loads(payload_raw)
    except Exception:
        return None

    now = int(time.time())
    if int(body.get("exp", 0)) < now:
        return None

    return body


def decode_access_token(token: str, *, secret: str) -> Optional[Dict[str, Any]]:
    body = _verify_token(token, secret=secret)
    if not body:
        return None
    if body.get("type") != "access":
        return None
    return body


def decode_refresh_token(token: str, *, secret: str) -> Optional[Dict[str, Any]]:
    body = _verify_token(token, secret=secret)
    if not body:
        return None
    if body.get("type") != "refresh":
        return None
    return body

