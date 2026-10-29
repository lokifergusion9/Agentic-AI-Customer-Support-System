"""Fixtures for attempts in a prepared eval environment: the three mocks in-process and a
factory that prepares a task and runs an adapter on it (W4.6, W4.7)."""

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.services.providers import ProviderClient
from evals.adapters.base import AgentAdapter, AgentResult, AttemptContext
from evals.environment import MockClients, prepare
from evals.loader import load_dataset
from evals.reviewer_simulator import ReviewerSimulator
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from mock_services.shipping.main import app as shipping_app
from mock_services.shipping.main import settings as shipping_settings
from mock_services.ticketing.main import app as ticketing_app
from mock_services.ticketing.main import settings as ticketing_settings
from tests.fakes import FakeEmbedder

TASKS = {task.id: task for task in load_dataset("support-v1")}
SCRIPTED_CONFIG = {"id": "scripted", "adapter": "scripted"}


@dataclass
class Attempt:
    result: AgentResult
    db: Session
    mocks: MockClients

    @property
    def run_id(self) -> str:
        return self.result.metadata["workflow_run_id"]

    def statuses(self) -> tuple[str, str]:
        return tuple(
            self.db.execute(
                text(
                    "SELECT r.status, t.status FROM support.workflow_runs r "
                    "JOIN support.tickets t ON t.id = r.ticket_id WHERE r.id = :r"
                ),
                {"r": self.run_id},
            ).one()
        )

    def events(self, event_type: str) -> list[dict]:
        return list(
            self.db.execute(
                text(
                    "SELECT redacted_payload FROM support.audit_events "
                    "WHERE run_id = :r AND event_type = :t ORDER BY sequence_number"
                ),
                {"r": self.run_id, "t": event_type},
            ).scalars()
        )

    def refunds(self, order_id: str) -> list[dict]:
        return self.mocks.payments.get("/refunds", params={"order_id": order_id}).json()


@pytest.fixture
def mocks() -> Iterator[MockClients]:
    with (
        TestClient(
            payments_app, headers={"X-API-Key": payments_settings.mock_payments_api_key}
        ) as pay,
        TestClient(
            shipping_app, headers={"X-API-Key": shipping_settings.mock_shipping_api_key}
        ) as ship,
        TestClient(
            ticketing_app, headers={"X-API-Key": ticketing_settings.mock_ticketing_api_key}
        ) as tick,
    ):
        yield MockClients(payments=pay, shipping=ship, ticketing=tick)


@pytest.fixture
def run_attempt(conn: Connection, mocks: MockClients):
    """run_attempt(task_id, adapter, agent_config=..., limits=..., cancelled=False)."""

    def run(task_id: str, adapter: AgentAdapter, **options) -> Attempt:
        task = TASKS[task_id]
        env = prepare(conn, mocks, task.evaluator, timeout_fault_seconds=0.01)
        db = Session(bind=conn, join_transaction_mode="create_savepoint")
        context = AttemptContext(
            db=db,
            environment=env,
            payments=ProviderClient("payment", mocks.payments),
            shipping=ProviderClient("shipping", mocks.shipping),
            ticketing=ProviderClient("ticketing", mocks.ticketing),
            embedder=FakeEmbedder(),
            reviewer=ReviewerSimulator(task.evaluator.reviewer_behavior),
            limits=options.get("limits", task.evaluator.evaluation.limits),
            agent_config=options.get("agent_config", SCRIPTED_CONFIG),
            sleep=lambda seconds: None,
        )
        if options.get("cancelled"):
            context.cancel.set()
        return Attempt(asyncio.run(adapter.run(task.agent, context)), db, mocks)

    return run
