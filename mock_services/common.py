"""Pieces shared by the mock services: structured errors, the API key check and fault queues.

Failures use the structured shape from W1.13. Every endpoint except `/health` needs the
`X-API-Key` header, so provider credentials stay on the server and are never something the
model can supply (PDF section 8).
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, FastAPI, Header, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class MockError(Exception):
    def __init__(
        self, status_code: int, error_code: str, message: str, *, retryable: bool = False
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.retryable = retryable


def add_error_handler(app: FastAPI) -> None:
    @app.exception_handler(MockError)
    def mock_error(_: Request, error: MockError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "success": False,
                "error_code": error.error_code,
                "retryable": error.retryable,
                "message": error.message,
            },
        )


def api_key_check(expected: str) -> Callable[..., None]:
    """A dependency that rejects requests without `X-API-Key: <expected>`."""

    def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if x_api_key != expected:
            raise MockError(status.HTTP_401_UNAUTHORIZED, "INVALID_API_KEY", "Invalid API key.")

    return require_api_key


@dataclass(frozen=True)
class Fault:
    """An injected failure for one request (see POST /admin/faults)."""

    mode: str
    status_code: int
    error_code: str
    retryable: bool
    delay_seconds: float


class FaultQueue:
    """Faults applied to the next requests, oldest first. Safe to use from any thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._faults: list[Fault] = []

    def add(self, fault: Fault, count: int) -> int:
        with self._lock:
            self._faults.extend([fault] * count)
            return len(self._faults)

    def pop(self) -> Fault | None:
        with self._lock:
            return self._faults.pop(0) if self._faults else None

    def clear(self) -> None:
        with self._lock:
            self._faults.clear()


class FaultRequestBase(BaseModel):
    """Each mock subclasses this and adds `mode: Literal[...]` with the modes it supports."""

    mode: str
    count: int = Field(default=1, ge=1, le=100)
    # mode=error: nothing is recorded and this error is returned.
    status_code: int = Field(default=503, ge=400, le=599)
    error_code: str = "PROVIDER_UNAVAILABLE"
    retryable: bool = True
    # Delay modes: how long to hold the response.
    delay_seconds: float = Field(default=5.0, ge=0, le=60)

    def to_fault(self) -> Fault:
        return Fault(
            mode=self.mode,
            status_code=self.status_code,
            error_code=self.error_code,
            retryable=self.retryable,
            delay_seconds=self.delay_seconds,
        )


def add_fault_routes(
    router: APIRouter, faults: FaultQueue, request_model: type[FaultRequestBase]
) -> None:
    @router.post("/admin/faults")
    def add_faults(request: request_model) -> dict[str, int]:  # type: ignore[valid-type]
        return {"queued": faults.add(request.to_fault(), request.count)}

    @router.delete("/admin/faults")
    def clear_faults() -> dict[str, int]:
        faults.clear()
        return {"queued": 0}


def raise_fault_error(fault: Fault) -> None:
    raise MockError(
        fault.status_code, fault.error_code, "Injected failure.", retryable=fault.retryable
    )
