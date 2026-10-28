"""Worker mode end to end: the API queues, the worker drives the run (W3.13)."""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, Engine, text
from sqlalchemy.orm import Session

from app.api.runs import get_dispatcher
from app.db.session import get_db
from app.jobs import queue
from app.jobs.handlers import HANDLERS, HandlerContext
from app.jobs.queue import LeaseLost
from app.jobs.worker import WorkerConfig, process_next, sweep
from app.main import app
from app.services.tickets import get_ticket
from app.workflows.dispatch import WorkerDispatcher
from app.workflows.runner import WorkflowServices, begin_resolution
from tests.integration import test_workflow as workflow
from tests.integration.test_proposals import ALICE
from tests.integration.test_runs_api import ALICE as ALICE_TOKEN
from tests.integration.test_runs_api import REVIEWER, open_ticket
from tests.integration.test_workflow import SPECIALIST, headphones_scenario, refund_ids

# The workflow tests' fixtures: seeded database, mock providers, scripted services.
db = workflow.db
providers = workflow.providers
make_services = workflow.make_services

# Long enough that the heartbeat thread never touches the test's shared connection.
CONFIG = WorkerConfig(worker_id="worker-test", lease_seconds=600)


@pytest.fixture
def services(db: Session, make_services, providers) -> WorkflowServices:
    payments, _ = providers
    return make_services(headphones_scenario(*refund_ids(db, payments)))


@pytest.fixture
def sessions(conn: Connection) -> Callable[[], Session]:
    """Worker sessions on the test's connection, so they see its uncommitted seed data."""
    return lambda: Session(bind=conn, join_transaction_mode="create_savepoint")


@pytest.fixture
def ctx(services: WorkflowServices, engine: Engine) -> HandlerContext:
    return HandlerContext(services, engine)


@pytest.fixture
def client(db: Session) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_dispatcher] = lambda: WorkerDispatcher(SPECIALIST)
    yield TestClient(app)
    app.dependency_overrides.clear()


def drain(sessions, ctx: HandlerContext) -> int:
    done = 0
    while process_next(sessions, ctx, CONFIG):
        done += 1
    return done


def jobs(db: Session) -> list[tuple[str, str]]:
    return [
        (row.payload["action"], row.status)
        for row in db.execute(text("SELECT payload, status FROM support.jobs ORDER BY created_at"))
    ]


def test_workflow_run_handler_is_registered() -> None:
    assert "workflow_run" in HANDLERS


def test_start_is_accepted_and_the_worker_runs_it(
    client: TestClient, db: Session, sessions, ctx: HandlerContext
) -> None:
    response = client.post(f"/tickets/{open_ticket(client)}/runs", headers=ALICE_TOKEN)

    assert response.status_code == 202
    run = response.json()
    assert (run["status"], run["output"]) == ("running", None)
    assert jobs(db) == [("drive", "queued")]
    assert client.get(f"/runs/{run['id']}", headers=ALICE_TOKEN).json()["status"] == "running"

    assert drain(sessions, ctx) == 1

    polled = client.get(f"/runs/{run['id']}", headers=ALICE_TOKEN).json()
    assert polled["status"] == "waiting_for_approval"
    assert jobs(db) == [("drive", "completed")]


def test_decision_is_accepted_and_the_worker_pays_once(
    client: TestClient, db: Session, sessions, ctx: HandlerContext, providers
) -> None:
    payments, _ = providers
    run_id = client.post(f"/tickets/{open_ticket(client)}/runs", headers=ALICE_TOKEN).json()["id"]
    drain(sessions, ctx)
    approval_id = client.get(f"/runs/{run_id}", headers=ALICE_TOKEN).json()["output"]["approval_id"]

    response = client.post(
        f"/approvals/{approval_id}/decision", json={"decision": "approved"}, headers=REVIEWER
    )

    assert response.status_code == 202
    assert response.json()["approval"]["decision"] == "approved"
    assert payments.get("/payments/ord_1001")["refunds"] == []  # nothing paid in the request
    drain(sessions, ctx)
    done = client.get(f"/runs/{run_id}", headers=ALICE_TOKEN).json()
    assert (done["status"], done["output"]["route"]) == ("completed", "refunded")
    assert len(payments.get("/payments/ord_1001")["refunds"]) == 1


def test_a_killed_worker_is_replaced_and_cannot_finish_the_job(
    client: TestClient, db: Session, sessions, ctx: HandlerContext
) -> None:
    run_id = client.post(f"/tickets/{open_ticket(client)}/runs", headers=ALICE_TOKEN).json()["id"]
    with sessions() as session:
        dead = queue.claim(session, "worker-dead", lease_seconds=60)  # claims, then dies
    assert dead is not None

    with sessions() as session:
        assert queue.requeue_expired(session, now=datetime.now(UTC) + timedelta(minutes=5))
    assert drain(sessions, ctx) == 1

    assert client.get(f"/runs/{run_id}", headers=ALICE_TOKEN).json()["status"] == (
        "waiting_for_approval"
    )
    with sessions() as session, pytest.raises(LeaseLost):
        queue.complete(session, dead)


def test_sweep_queues_a_run_left_without_a_job(db: Session, sessions, ctx: HandlerContext) -> None:
    # The API created the run but stopped before queueing its job.
    ticket_id = workflow.open_ticket(db, workflow.HEADPHONES, "ord_1001")
    run = begin_resolution(db, ALICE, ticket_id, SPECIALIST)

    sweep(sessions, ctx)

    assert ("drive", "queued") in jobs(db)
    drain(sessions, ctx)
    assert get_ticket(db, ALICE, run.ticket_id).status == "awaiting_approval"


def test_sweep_does_not_queue_a_second_job_for_a_run(
    client: TestClient, db: Session, sessions, ctx: HandlerContext
) -> None:
    client.post(f"/tickets/{open_ticket(client)}/runs", headers=ALICE_TOKEN)

    sweep(sessions, ctx)
    sweep(sessions, ctx)

    assert jobs(db) == [("drive", "queued")]
