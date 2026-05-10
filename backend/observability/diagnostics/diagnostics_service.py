from __future__ import annotations

import os
import platform
from typing import Any, Dict


def get_system_status() -> Dict[str, Any]:
    return {
        "status": "running",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "environment": os.getenv("ENVIRONMENT", "development"),
    }

