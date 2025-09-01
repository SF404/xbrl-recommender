import traceback
import uuid
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError

from .config import get_settings


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    DB_ERROR = "DB_ERROR"
    SETTINGS_NOT_FOUND = "SETTINGS_NOT_FOUND"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INDEX_NOT_FOUND = "INDEX_NOT_FOUND"
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    FILE_VALIDATION_ERROR = "FILE_VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


class AppException(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int = 400, detail: Optional[Any] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = detail


def _error_payload(code: ErrorCode, message: str, request_id: str, status_code: int, detail: Any = None):
    settings = get_settings()
    payload: Dict[str, Any] = {
        "error": {
            "id": str(uuid.uuid4()),
            "code": code,
            "message": message,
        },
        "meta": {"request_id": request_id},
    }
    if settings.is_dev:
        payload["error"]["detail"] = detail
        payload["error"]["stack"] = traceback.format_exc().splitlines()
    return JSONResponse(status_code=status_code, content=payload)


def install_exception_handlers(app: FastAPI):
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        req_id = request.state.request_id if hasattr(request.state, "request_id") else "n/a"
        return _error_payload(exc.code, exc.message, req_id, exc.status_code, detail=exc.detail)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = request.state.request_id if hasattr(request.state, "request_id") else "n/a"
        return _error_payload(ErrorCode.VALIDATION_ERROR, "Validation error", req_id, 422, detail=exc.errors())

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        req_id = request.state.request_id if hasattr(request.state, "request_id") else "n/a"
        return _error_payload(ErrorCode.HTTP_ERROR, exc.detail, req_id, exc.status_code)

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
        req_id = request.state.request_id if hasattr(request.state, "request_id") else "n/a"
        return _error_payload(ErrorCode.DB_ERROR, "Database error", req_id, 500, detail=str(exc))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        req_id = request.state.request_id if hasattr(request.state, "request_id") else "n/a"
        return _error_payload(ErrorCode.INTERNAL_SERVER_ERROR, "Unexpected server error", req_id, 500, detail=str(exc))
