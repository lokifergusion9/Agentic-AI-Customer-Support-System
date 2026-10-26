"""Failures raised by services, carrying the structured shape from W1.13."""

from app.schemas.errors import ErrorResponse


class ServiceFailure(Exception):
    def __init__(self, error_code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.retryable = retryable

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            error_code=self.error_code, retryable=self.retryable, message=self.message
        )
