"""Request parsing and validation formatting helpers for API routes."""

from typing import Any, Dict, List

from fastapi import Request


def sensor_activity_label(activity_level: int) -> str:
    """Map the UI's 1-10 activity slider to inference labels."""
    if activity_level <= 3:
        return "low"
    if activity_level >= 8:
        return "high"
    return "moderate"


def payload_dict(model: Any) -> Dict[str, Any]:
    """Return a Pydantic model as a plain dict across v1/v2 APIs."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def validation_field(loc: List[Any]) -> str:
    """Convert a Pydantic validation location into a user-facing field name."""
    parts = [
        str(part)
        for part in loc
        if str(part) not in {"body", "query", "path"}
    ]
    return ".".join(parts) if parts else "request"


def format_validation_error(error: Dict[str, Any]) -> str:
    """Format one Pydantic validation error without changing response shape."""
    field = validation_field(error.get("loc", []))
    error_type = error.get("type", "")
    message = error.get("msg", "Invalid value")
    ctx = error.get("ctx") or {}

    if error_type == "missing":
        return f"Missing required field: {field}"
    if error_type == "extra_forbidden":
        return f"Unexpected field: {field}"
    if "enum" in error_type:
        expected = ctx.get("expected")
        suffix = f": expected {expected}" if expected else ""
        return f"Invalid {field} value{suffix}"
    if error_type == "greater_than_equal":
        return f"Invalid {field} value: must be at least {ctx.get('ge')}"
    if error_type == "less_than_equal":
        return f"Invalid {field} value: must be at most {ctx.get('le')}"
    if error_type in {"int_parsing", "int_type"}:
        return f"Invalid {field} value: must be an integer"
    if error_type in {"float_parsing", "float_type"}:
        return f"Invalid {field} value: must be a number"
    if error_type == "string_too_short":
        return f"Invalid {field} value: must be at least {ctx.get('min_length')} characters"
    if error_type == "string_too_long":
        return f"Invalid {field} value: must be at most {ctx.get('max_length')} characters"

    return f"Invalid {field} value: {message}"


def format_validation_errors(errors: List[Dict[str, Any]]) -> str:
    """Join Pydantic validation errors into the existing frontend-readable string."""
    messages = [format_validation_error(error) for error in errors]
    return "; ".join(messages) if messages else "Invalid request payload"


async def read_request_payload(request: Request) -> Any:
    """Best-effort request payload read for diagnostics without raising."""
    try:
        return await request.json()
    except Exception:
        try:
            body = await request.body()
            return body.decode("utf-8", errors="replace")
        except Exception as exc:
            return f"<unavailable: {exc}>"
