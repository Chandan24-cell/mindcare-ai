from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class UserProfile(BaseModel):
    user_id: str = Field(..., min_length=2, max_length=64)
    emotional_preferences: Dict[str, Any] = Field(default_factory=dict)
    conversation_preferences: Dict[str, Any] = Field(default_factory=dict)
    coping_preferences: Dict[str, Any] = Field(default_factory=dict)

