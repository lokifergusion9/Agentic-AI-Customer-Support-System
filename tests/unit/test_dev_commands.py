"""The developer commands never aim an experiment at the application database (W5.8)."""

import pytest

from scripts import dev


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("postgresql+psycopg://u:p@localhost:5432/support", "support_eval"),
        ("postgresql+psycopg://u:p@localhost:5432/support_eval", "support_eval"),
        ("postgresql+psycopg://u:p@localhost:5432/shop", "shop_eval"),
    ],
)
def test_the_eval_database_is_named_after_the_application_database(
    monkeypatch: pytest.MonkeyPatch, configured: str, expected: str
) -> None:
    monkeypatch.setattr(dev, "app_url", lambda: dev.make_url(configured))
    url = dev.eval_url()
    assert url.database == expected
    assert (url.username, url.host, url.port) == ("u", "localhost", 5432)


def test_a_child_process_gets_the_database_it_is_pointed_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}

    def fake_run(command, check, cwd, env):
        seen.update(env)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost/support")
    monkeypatch.setattr(dev.subprocess, "run", fake_run)
    target = dev.make_url("postgresql+psycopg://u:p@localhost/support_eval")
    dev.run(["true"], target)
    assert seen["DATABASE_URL"].endswith("/support_eval")


def test_eval_mock_clients_time_out_like_the_application_does() -> None:
    """A timeout fault holds a response one second past the application's provider timeout.

    The eval clients once had a longer timeout (10 s against 5 s), so the fault never looked
    like a timeout: the refund came back normally and nothing was reconciled.
    """
    from app.config import get_settings
    from evals.run_evals import mock_clients

    mocks = mock_clients()
    expected = get_settings().provider_timeout_seconds
    for client in (mocks.payments, mocks.shipping, mocks.ticketing):
        assert client.timeout.read == expected
