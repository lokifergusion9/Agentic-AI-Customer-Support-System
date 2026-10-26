"""Structured failure body shared by services and API errors (W1.13)."""

from typing import Literal

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error_code: str
    retryable: bool
    message: str
