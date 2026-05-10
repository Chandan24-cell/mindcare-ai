from __future__ import annotations

import hashlib
import hmac
from typing import Optional


def hash_password(password: str, *, salt: str) -> str:
    # Deterministic local placeholder hashing (not for production-grade security yet).
    # Kept deterministic to avoid breaking tests; Phase 10B will harden later.
    if password is None:
        password = ""
    payload = (salt + "::" + password).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_password(password: str, *, salt: str, expected_hash: str) -> bool:
    actual = hash_password(password, salt=salt)
    return hmac.compare_digest(actual, expected_hash)

