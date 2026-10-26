"""Provider client: every failure becomes a structured ServiceFailure (W1.12, W1.13)."""

from datetime import date

import httpx
import pytest
from pydantic import BaseModel

from app.config import get_settings
from app.services.errors import ServiceFailure
from app.services.providers import (
    ProviderClient,
    get_payments_client,
    get_shipping_client,
    get_ticketing_client,
    parse,
)
from mock_services.shipping.main import app as shipping_app
from mock_services.shipping.main import settings as shipping_settings
from scripts.seed_data import push_mock_shipping
from tests.unit.live_server import serve


def client(handler) -> ProviderClient:
    return ProviderClient(
        "shipping", httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
    )


def raising(error: Exception):
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    return handler


def failure(provider: ProviderClient, path: str = "/anything") -> ServiceFailure:
    with pytest.raises(ServiceFailure) as error:
        provider.get(path)
    return error.value


def test_success_returns_the_json_body() -> None:
    provider = client(lambda request: httpx.Response(200, json={"status": "ok"}))
    assert provider.get("/anything") == {"status": "ok"}


def test_request_passes_method_params_and_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={})

    client(handler).request("POST", "/refunds", params={"a": "1"}, json={"b": 2})
    assert (seen[0].method, seen[0].url.path, seen[0].url.params["a"]) == ("POST", "/refunds", "1")
    assert seen[0].content == b'{"b":2}'


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (httpx.ReadTimeout("slow"), "PROVIDER_TIMEOUT"),
        (httpx.ConnectTimeout("slow"), "PROVIDER_TIMEOUT"),
        (httpx.ConnectError("refused"), "PROVIDER_UNAVAILABLE"),
        (httpx.RemoteProtocolError("dropped"), "PROVIDER_UNAVAILABLE"),
    ],
)
def test_transport_errors_are_retryable(error: Exception, code: str) -> None:
    result = failure(client(raising(error)))
    assert (result.error_code, result.retryable) == (code, True)
    assert "shipping" in result.message


@pytest.mark.parametrize("retryable", [True, False])
def test_structured_provider_errors_pass_through(retryable: bool) -> None:
    body = {"success": False, "error_code": "COURIER_DOWN", "retryable": retryable, "message": "m"}
    result = failure(client(lambda request: httpx.Response(503, json=body)))
    assert (result.error_code, result.retryable, result.message) == ("COURIER_DOWN", retryable, "m")


@pytest.mark.parametrize(
    ("response", "retryable"),
    [
        (httpx.Response(502, text="<html>Bad gateway</html>"), True),
        (httpx.Response(500, json={"detail": "oops"}), True),
        (httpx.Response(400, json={"detail": "bad"}), False),
        (httpx.Response(404, text="not here"), False),
    ],
    ids=["html_502", "json_500", "json_400", "text_404"],
)
def test_other_error_responses_are_provider_errors(
    response: httpx.Response, retryable: bool
) -> None:
    result = failure(client(lambda request: response))
    assert (result.error_code, result.retryable) == ("PROVIDER_ERROR", retryable)
    assert str(response.status_code) in result.message


class Thing(BaseModel):
    id: str
    amount_minor: int


def test_parse_validates_the_response() -> None:
    assert parse(Thing, {"id": "x", "amount_minor": 5}, "payment") == Thing(id="x", amount_minor=5)


@pytest.mark.parametrize("data", [{"id": "x"}, {"id": "x", "amount_minor": "lots"}, [], None])
def test_parse_turns_a_bad_shape_into_a_provider_error(data: object) -> None:
    with pytest.raises(ServiceFailure) as error:
        parse(Thing, data, "payment")
    assert (error.value.error_code, error.value.retryable) == ("PROVIDER_ERROR", False)


@pytest.mark.parametrize(
    ("dependency", "url", "key"),
    [
        (get_payments_client, "mock_payments_url", "mock_payments_api_key"),
        (get_shipping_client, "mock_shipping_url", "mock_shipping_api_key"),
        (get_ticketing_client, "mock_ticketing_url", "mock_ticketing_api_key"),
    ],
)
def test_dependencies_use_the_settings(dependency, url: str, key: str) -> None:
    settings = get_settings()
    provider = dependency()
    assert provider is dependency()  # one client and connection pool per provider
    assert str(provider.http.base_url).rstrip("/") == getattr(settings, url)
    assert provider.http.headers["X-API-Key"] == getattr(settings, key)
    assert provider.http.timeout.read == settings.provider_timeout_seconds


def test_real_timeout_against_a_slow_provider() -> None:
    headers = {"X-API-Key": shipping_settings.mock_shipping_api_key}
    with serve(shipping_app) as url, httpx.Client(base_url=url, headers=headers) as admin:
        push_mock_shipping(admin, date(2026, 10, 25))
        admin.post("/admin/faults", json={"mode": "delay", "delay_seconds": 1})

        provider = ProviderClient(
            "shipping", httpx.Client(base_url=url, headers=headers, timeout=0.3)
        )
        result = failure(provider, "/shipments/trk_1001")
        assert (result.error_code, result.retryable) == ("PROVIDER_TIMEOUT", True)
        assert provider.get("/shipments/trk_1001")["status"] == "delivered"
