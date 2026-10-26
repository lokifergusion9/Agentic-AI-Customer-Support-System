"""HTTP clients for the external providers: payments, shipping and ticketing (W1.12).

Every failure becomes a ServiceFailure in the W1.13 shape, so a service never leaks an httpx
exception and the agent always gets an error code and a retryable flag:

- timeout                    -> PROVIDER_TIMEOUT, retryable
- connection error           -> PROVIDER_UNAVAILABLE, retryable
- structured provider error  -> the provider's own error_code and retryable flag
- any other error response   -> PROVIDER_ERROR, retryable only for 5xx
- a body of the wrong shape  -> PROVIDER_ERROR, not retryable (see `parse`)

A timeout on a write does not mean the write failed; the caller reconciles (W3.12).
"""

from functools import lru_cache
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.services.errors import ServiceFailure


def parse[M: BaseModel](model: type[M], data: Any, name: str) -> M:
    """Validate a provider response, so a changed or broken provider fails cleanly."""
    try:
        return model.model_validate(data)
    except ValidationError:
        raise ServiceFailure(
            "PROVIDER_ERROR", f"The {name} service returned an unexpected response."
        ) from None


class ProviderClient:
    def __init__(self, name: str, http: httpx.Client) -> None:
        self.name = name
        self.http = http

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send a request and return the decoded JSON body, or raise ServiceFailure."""
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.TimeoutException:
            raise ServiceFailure(
                "PROVIDER_TIMEOUT",
                f"The {self.name} service did not respond in time.",
                retryable=True,
            ) from None
        except httpx.TransportError:
            raise ServiceFailure(
                "PROVIDER_UNAVAILABLE", f"The {self.name} service is not reachable.", retryable=True
            ) from None
        if response.is_success:
            return response.json()
        raise provider_failure(self.name, response)

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)


def provider_failure(name: str, response: httpx.Response) -> ServiceFailure:
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("success") is False and "error_code" in body:
        return ServiceFailure(
            str(body["error_code"]),
            str(body.get("message", "")),
            retryable=bool(body.get("retryable", False)),
        )
    return ServiceFailure(
        "PROVIDER_ERROR",
        f"The {name} service returned an unexpected error ({response.status_code}).",
        retryable=response.status_code >= 500,
    )


def provider_client(name: str, base_url: str, api_key: str) -> ProviderClient:
    http = httpx.Client(
        base_url=base_url,
        headers={"X-API-Key": api_key},
        timeout=get_settings().provider_timeout_seconds,
    )
    return ProviderClient(name, http)


# FastAPI dependencies; one client (and connection pool) per provider. Tests override them.


@lru_cache
def get_payments_client() -> ProviderClient:
    settings = get_settings()
    return provider_client("payment", settings.mock_payments_url, settings.mock_payments_api_key)


@lru_cache
def get_shipping_client() -> ProviderClient:
    settings = get_settings()
    return provider_client("shipping", settings.mock_shipping_url, settings.mock_shipping_api_key)


@lru_cache
def get_ticketing_client() -> ProviderClient:
    settings = get_settings()
    return provider_client(
        "ticketing", settings.mock_ticketing_url, settings.mock_ticketing_api_key
    )
