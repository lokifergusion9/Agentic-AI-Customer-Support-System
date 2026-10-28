"""Concurrent refund execution never over-reserves or pays twice (W3.10).

Unlike most integration tests these commit, so separate sessions in separate threads see the
same rows and really contend for the order's row lock. The tables are truncated afterwards.
"""

import threading
from collections.abc import Iterator

import httpx
import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from app.db import SCHEMA
from app.services.errors import ServiceFailure
from app.services.providers import ProviderClient
from app.services.refunds import RefundOperationView, execute_approved_refund
from mock_services.payments.main import app as payments_app
from scripts.seed_data import push_mock_payments, seed
from tests.integration.test_proposals import ALICE, TODAY
from tests.integration.test_refund_execution import HEADERS, approved
from tests.unit.live_server import serve


@pytest.fixture
def committed(engine: Engine) -> Iterator[Engine]:
    with engine.begin() as connection:
        seed(connection, TODAY)
    yield engine
    with engine.begin() as cleanup:
        # TRUNCATE skips row-level triggers, so the append-only audit table is cleared too.
        cleanup.execute(
            text(
                f"TRUNCATE {SCHEMA}.customers, {SCHEMA}.policy_documents, "
                f"{SCHEMA}.audit_events CASCADE"
            )
        )


@pytest.fixture
def provider_url() -> Iterator[str]:
    with serve(payments_app) as url:
        with httpx.Client(base_url=url, headers=HEADERS) as admin:
            push_mock_payments(admin, TODAY)
        yield url


def client(url: str) -> ProviderClient:
    return ProviderClient("payment", httpx.Client(base_url=url, headers=HEADERS, timeout=5))


def race(engine: Engine, url: str, proposal_ids: list[str]) -> list[object]:
    """Execute each proposal in its own thread and session, all released at once."""
    start = threading.Barrier(len(proposal_ids))
    results: list[object] = [None] * len(proposal_ids)

    def worker(index: int, proposal_id: str) -> None:
        with Session(engine) as db:
            start.wait()
            try:
                results[index] = execute_approved_refund(db, client(url), proposal_id)
            except ServiceFailure as failure:
                results[index] = failure

    threads = [threading.Thread(target=worker, args=(i, pid)) for i, pid in enumerate(proposal_ids)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return results


def approve_twice(engine: Engine, url: str) -> list[str]:
    with Session(engine) as db:
        payments = client(url)
        # Two tickets for the same order, each with its own approved full refund.
        return [approved(db, payments, ALICE, "ord_1001").proposal_id for _ in range(2)]


def test_two_approved_refunds_for_one_order_reserve_the_balance_once(
    committed: Engine, provider_url: str
) -> None:
    proposals = approve_twice(committed, provider_url)

    results = race(committed, provider_url, proposals)

    paid = [r for r in results if isinstance(r, RefundOperationView)]
    refused = [r for r in results if isinstance(r, ServiceFailure)]
    assert [op.status for op in paid] == ["succeeded"]
    assert [f.error_code for f in refused] in (["REFUND_EXCEEDS_BALANCE"], ["NOT_ELIGIBLE"])
    assert len(client(provider_url).get("/refunds", params={"order_id": "ord_1001"})) == 1
    with committed.connect() as connection:
        amounts = connection.execute(
            text(f"SELECT amount_minor FROM {SCHEMA}.refund_operations WHERE order_id = 'ord_1001'")
        ).scalars()
        assert list(amounts) == [249900]


def test_one_proposal_executed_twice_at_once_pays_once(
    committed: Engine, provider_url: str
) -> None:
    [proposal_id, _] = approve_twice(committed, provider_url)

    results = race(committed, provider_url, [proposal_id, proposal_id])

    assert all(isinstance(r, RefundOperationView) for r in results), results
    operations = {r.operation_id for r in results}  # type: ignore[union-attr]
    assert len(operations) == 1
    assert len(client(provider_url).get("/refunds", params={"order_id": "ord_1001"})) == 1
