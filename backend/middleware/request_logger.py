from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Keep lightweight; existing app already logs via logger.
        response = await call_next(request)
        return response

