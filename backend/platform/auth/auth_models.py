from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class RegisterRequest(BaseModel):
    user_id: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=6, max_length=200)


class LoginRequest(BaseModel):
    user_id: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=6, max_length=200)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10)


class LogoutRequest(BaseModel):
    access_token: str = Field(..., min_length=10)

