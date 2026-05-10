from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.api.request_utils import format_validation_errors, read_request_payload


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    errors = exc.errors()
    readable_error = format_validation_errors(errors)
    payload = await read_request_payload(request)

    # Preserve existing special-case behavior.
    if request.url.path == "/predict/sensor":
        # keep logging responsibility to app/main logger; do not change payload contract
        pass

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": readable_error,
            "detail": readable_error,
            "validation_errors": jsonable_encoder(errors),
        },
    )

