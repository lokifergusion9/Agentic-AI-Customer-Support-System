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


@dataclass(frozen=True)
class KeyedFault:
    """A failure for one logical operation, from its `occurrence`-th matching call on (E§22).

    Matching by operation and attributes (an order ID, say) instead of by request position
    means two agents that call the provider in a different order still meet the same failure.
    """

    operation: str
    match: tuple[tuple[str, str], ...]
    occurrence: int
    count: int
    fault: Fault

    def matches(self, operation: str, attributes: dict[str, str]) -> bool:
        return operation == self.operation and all(
            attributes.get(key) == value for key, value in self.match
        )


class FaultQueue:
    """Injected faults, safe to use from any thread.

    Two kinds: keyed faults fire on a logical operation (see KeyedFault); the plain queue
    applies to the next requests of the operations that consume it, oldest first.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._faults: list[Fault] = []
        self._keyed: list[KeyedFault] = []
        self._seen: list[int] = []  # matching calls so far, per keyed fault

    def add(self, fault: Fault, count: int) -> int:
        with self._lock:
            self._faults.extend([fault] * count)
            return len(self._faults) + len(self._keyed)

    def add_keyed(self, fault: KeyedFault) -> int:
        with self._lock:
            self._keyed.append(fault)
            self._seen.append(0)
            return len(self._faults) + len(self._keyed)

    def pop(self) -> Fault | None:
        with self._lock:
            return self._faults.pop(0) if self._faults else None

    def take(self, operation: str, *, queued: bool, **attributes: str) -> Fault | None:
        """The fault for this call of `operation`, if any.

        Every matching keyed fault counts the call, whether or not it fires. `queued` says
        whether this operation also consumes the plain queue (the ones that did before keyed
        faults existed), so old tests keep their meaning.
        """
        with self._lock:
            hit = None
            for index, keyed in enumerate(self._keyed):
                if not keyed.matches(operation, attributes):
                    continue
                self._seen[index] += 1
                if hit is None and keyed.occurrence <= self._seen[index] < (
                    keyed.occurrence + keyed.count
                ):
                    hit = keyed.fault
            if hit is not None:
                return hit
            return self._faults.pop(0) if queued and self._faults else None

    def clear(self) -> None:
        with self._lock:
            self._faults.clear()
            self._keyed.clear()
            self._seen.clear()


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
    # Keyed faults: set `operation` to fail that operation from its `occurrence`-th matching
    # call, for `count` calls. `match` narrows it, e.g. {"order_id": "ord_1042"}. Without
    # `operation`, the fault goes on the plain queue for the next `count` requests.
    operation: str | None = None
    match: dict[str, str] = {}
    occurrence: int = Field(default=1, ge=1, le=100)

    def to_fault(self) -> Fault:
        return Fault(
            mode=self.mode,
            status_code=self.status_code,
            error_code=self.error_code,
            retryable=self.retryable,
            delay_seconds=self.delay_seconds,
        )


def add_fault_routes(
    router: APIRouter,
    faults: FaultQueue,
    request_model: type[FaultRequestBase],
    operations: frozenset[str],
) -> None:
    """POST/DELETE /admin/faults. `operations` lists the logical operations this mock fails."""

    @router.post("/admin/faults")
    def add_faults(request: request_model) -> dict[str, int]:  # type: ignore[valid-type]
        if request.operation is None:
            if request.match or request.occurrence != 1:
                raise MockError(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    "INVALID_FAULT",
                    "match and occurrence need an operation.",
                )
            return {"queued": faults.add(request.to_fault(), request.count)}
        if request.operation not in operations:
            raise MockError(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "UNKNOWN_OPERATION",
                f"Operations: {', '.join(sorted(operations))}.",
            )
        keyed = KeyedFault(
            operation=request.operation,
            match=tuple(sorted(request.match.items())),
            occurrence=request.occurrence,
            count=request.count,
            fault=request.to_fault(),
        )
        return {"queued": faults.add_keyed(keyed)}

    @router.delete("/admin/faults")
    def clear_faults() -> dict[str, int]:
        faults.clear()
        return {"queued": 0}


def raise_fault_error(fault: Fault) -> None:
    raise MockError(
        fault.status_code, fault.error_code, "Injected failure.", retryable=fault.retryable
    )
