"""Per-attempt environment: reset, fixture loading, faults, snapshot, trace and lock (W4.4, E§28)."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, Engine, text

from app.db import SCHEMA
from evals.db import EVAL_SCHEMA
from evals.environment import MockClients, capture, environment_lock, prepare
from evals.loader import load_dataset, split_views
from evals.task_format import EvaluatorView, Task
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from mock_services.shipping.main import app as shipping_app
from mock_services.shipping.main import settings as shipping_settings
from mock_services.ticketing.main import app as ticketing_app
from mock_services.ticketing.main import settings as ticketing_settings

TASKS = {task.id: task.evaluator for task in load_dataset("support-v1")}


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


def ids(conn: Connection, table: str) -> list[str]:
    return list(conn.execute(text(f"SELECT id FROM {SCHEMA}.{table} ORDER BY id")).scalars())


def new_attempt(conn: Connection, task: EvaluatorView, env_id: str) -> str:
    """The eval rows an attempt hangs from (normally written by W4.8)."""
    statements = [
        f"INSERT INTO {EVAL_SCHEMA}.datasets (id) VALUES ('support')",
        (
            f"INSERT INTO {EVAL_SCHEMA}.dataset_versions (id, dataset_id, version, content_hash) "
            "VALUES ('support-v1', 'support', 1, 'sha256:d')"
        ),
        (
            f"INSERT INTO {EVAL_SCHEMA}.task_versions (id, task_id, version, content_hash, category, "
            "difficulty, scenario_family, agent_view, evaluator_view) VALUES (:tv, :task, :version, "
            ":hash, :category, :difficulty, :family, '{}'::jsonb, CAST(:view AS jsonb))"
        ),
        (
            f"INSERT INTO {EVAL_SCHEMA}.agent_configs (id, adapter, provider, model_id, "
            "prompt_version, architecture, tool_schema_hash, config_hash) VALUES ('scripted', "
            "'scripted', 'none', 'none', 'v1', 'scripted', 'sha256:t', 'sha256:c')"
        ),
        (
            f"INSERT INTO {EVAL_SCHEMA}.experiments (id, name, idempotency_key, dataset_version_id, "
            "split, repetitions, manifest) VALUES ('exp_t', 't', 't', 'support-v1', 'dev', 1, "
            "'{}'::jsonb)"
        ),
        (
            f"INSERT INTO {EVAL_SCHEMA}.runs (id, experiment_id, agent_config_id, task_version_id, "
            "repetition) VALUES ('erun_t', 'exp_t', 'scripted', :tv, 1)"
        ),
        (
            f"INSERT INTO {EVAL_SCHEMA}.attempts (id, run_id, attempt_number, environment_id) "
            "VALUES ('att_t', 'erun_t', 1, :env)"
        ),
    ]
    params = {
        "tv": f"{task.task_id}@v{task.version}",
        "task": task.task_id,
        "version": task.version,
        "hash": task.content_hash,
        "category": task.category,
        "difficulty": task.difficulty,
        "family": task.scenario_family,
        "view": task.model_dump_json(),
        "env": env_id,
    }
    for statement in statements:
        conn.execute(text(statement), params)
    return "att_t"


def test_the_fixture_is_loaded_into_the_database_and_the_mocks(conn, mocks) -> None:
    env = prepare(conn, mocks, TASKS["refund_eligible_001"])

    assert env.environment_id.startswith("env_")
    assert str(env.as_of) == "2026-10-25"
    assert ids(conn, "customers") == ["cus_alice"]
    assert ids(conn, "orders") == ["ord_1001"]
    assert conn.execute(text(f"SELECT status, inspection_status FROM {SCHEMA}.returns")).one() == (
        "received",
        "passed",
    )
    payment = mocks.payments.get("/payments/ord_1001").json()
    assert (payment["amount_minor"], payment["refunds"]) == (249900, [])
    assert mocks.shipping.get("/shipments", params={"order_id": "ord_1001"}).json()
    assert mocks.payments.get("/payments/ord_1002").status_code == 404  # only this fixture


def test_a_past_refund_is_loaded_on_both_sides(conn, mocks) -> None:
    prepare(conn, mocks, TASKS["duplicate_refund_succeeded_001"])

    assert conn.execute(
        text(f"SELECT status, amount_minor FROM {SCHEMA}.refund_operations")
    ).one() == ("succeeded", 399900)
    refunds = mocks.payments.get("/refunds", params={"order_id": "ord_1002"}).json()
    assert [(r["status"], r["amount_minor"]) for r in refunds] == [("succeeded", 399900)]
    assert mocks.ticketing.get("/tickets/ext_1002").status_code == 200


def test_repeated_attempts_start_from_clean_state(conn, mocks) -> None:
    task = TASKS["refund_eligible_001"]
    prepare(conn, mocks, task)
    # What a first attempt leaves behind: a refund at the provider and a new ticket.
    refund = {"idempotency_key": "k", "order_id": "ord_1001", "amount_minor": 249900}
    assert mocks.payments.post("/refunds", json={**refund, "currency": "INR"}).is_success
    conn.execute(
        text(
            f"INSERT INTO {SCHEMA}.tickets (id, customer_id, order_id, category, status, "
            "customer_message) VALUES ('tkt_x', 'cus_alice', 'ord_1001', 'refund_request', "
            "'open', 'refund please')"
        )
    )

    prepare(conn, mocks, task)

    assert ids(conn, "tickets") == []
    assert mocks.payments.get("/refunds", params={"order_id": "ord_1001"}).json() == []


def test_an_attempt_cannot_see_the_previous_attempts_environment(conn, mocks) -> None:
    prepare(conn, mocks, TASKS["refund_high_value_001"])
    prepare(conn, mocks, TASKS["lookup_delivered_001"])

    assert ids(conn, "orders") == ["ord_1011"]
    assert ids(conn, "customers") == ["cus_dev"]
    assert mocks.payments.get("/payments/ord_1014").status_code == 404
    assert mocks.shipping.get("/shipments", params={"order_id": "ord_1014"}).json() == []


def test_the_fixture_is_not_mutated_by_loading() -> None:
    task = TASKS["duplicate_refund_succeeded_001"]
    before = task.model_dump_json()
    with pytest.raises(AttributeError):  # tuples: nothing to append to
        task.environment_fixture.orders.append(None)  # type: ignore[arg-type]
    assert task.model_dump_json() == before


def test_loaded_state_excludes_expected_outcomes(conn, mocks) -> None:
    task = TASKS["access_other_refund_001"]
    prepare(conn, mocks, task)

    visible = json.dumps(
        {
            table: [dict(r._mapping) for r in conn.execute(text(f"SELECT * FROM {SCHEMA}.{table}"))]
            for table in ("customers", "orders", "order_items", "returns", "tickets")
        },
        default=str,
    ) + json.dumps(mocks.payments.get("/payments/ord_1003").json())
    hidden = [
        "awaiting_customer",
        "refund_requires_valid_approval",
        *task.evaluation.response_checks.must_not_claim,
    ]
    assert [h for h in hidden if h in visible] == []
    assert conn.execute(text(f"SELECT count(*) FROM {EVAL_SCHEMA}.task_versions")).scalar() == 0


def test_jobs_survive_the_reset(conn, mocks) -> None:
    conn.execute(
        text(
            f"INSERT INTO {SCHEMA}.jobs (id, job_type, payload) "
            "VALUES ('job_eval', 'eval_attempt', '{}'::jsonb)"
        )
    )
    prepare(conn, mocks, TASKS["lookup_delivered_001"])
    assert ids(conn, "jobs") == ["job_eval"]


def test_faults_fire_on_the_named_operation_and_order(conn, mocks) -> None:
    prepare(conn, mocks, TASKS["fault_shipping_transient_001"])  # shipment.read, 5xx twice

    def read(order_id: str) -> int:
        return mocks.shipping.get("/shipments", params={"order_id": order_id}).status_code

    assert [read("ord_2171") for _ in range(3)] == [503, 503, 200]

    prepare(conn, mocks, TASKS["fault_payment_read_down_001"])  # payment.get, 5xx ten times
    assert mocks.payments.get("/payments/ord_2173").status_code == 503


def test_an_unknown_outcome_fault_records_the_refund_then_holds_the_response(conn, mocks) -> None:
    prepare(conn, mocks, TASKS["fault_refund_accept_timeout_001"], timeout_fault_seconds=0.01)
    refund = {"idempotency_key": "k", "order_id": "ord_1001", "amount_minor": 249900}
    assert mocks.payments.post("/refunds", json={**refund, "currency": "INR"}).is_success
    assert len(mocks.payments.get("/refunds", params={"order_id": "ord_1001"}).json()) == 1


def test_capture_stores_the_final_state_and_copies_the_trace(conn, mocks) -> None:
    task = TASKS["duplicate_refund_succeeded_001"]
    env = prepare(conn, mocks, task)
    attempt_id = new_attempt(conn, task, env.environment_id)
    # The seeded past refund brings ticket tkt_1002 and run run_1002; give them a trace.
    for run_id, event in [(None, "ticket_created"), ("run_1002", "run_started")]:
        conn.execute(
            text(
                f"INSERT INTO {SCHEMA}.audit_events (run_id, ticket_id, actor, event_type) "
                "VALUES (:run, 'tkt_1002', 'system:test', :event)"
            ),
            {"run": run_id, "event": event},
        )

    state = capture(conn, mocks, env, attempt_id, "run_1002")

    assert state["environment_id"] == env.environment_id
    assert [r["status"] for r in state["support"]["refund_operations"]] == ["succeeded"]
    assert [r["status"] for r in state["providers"]["refunds"]] == ["succeeded"]
    assert [t["ref"] for t in state["providers"]["tickets"]] == ["ext_1002"]
    stored = conn.execute(
        text(f"SELECT content, content_hash FROM {EVAL_SCHEMA}.artifacts WHERE attempt_id = :a"),
        {"a": attempt_id},
    ).one()
    assert stored.content == state and stored.content_hash.startswith("sha256:")
    trace = conn.execute(
        text(
            f"SELECT sequence_number, event_type FROM {EVAL_SCHEMA}.trace_events "
            "WHERE attempt_id = :a ORDER BY sequence_number"
        ),
        {"a": attempt_id},
    ).all()
    assert [tuple(row) for row in trace] == [(1, "ticket_created"), (2, "run_started")]

    # The copy outlives the next reset.
    prepare(conn, mocks, TASKS["lookup_delivered_001"])
    count = f"SELECT count(*) FROM {EVAL_SCHEMA}.trace_events WHERE attempt_id = :a"
    assert conn.execute(text(count), {"a": attempt_id}).scalar() == 2


def test_only_one_attempt_holds_the_environment(conn: Connection, engine: Engine) -> None:
    try_lock = "SELECT pg_try_advisory_lock(hashtextextended('eval:environment', 0))"
    with environment_lock(conn), engine.connect() as other:
        assert other.execute(text(try_lock)).scalar() is False
    with engine.connect() as other:
        assert other.execute(text(try_lock)).scalar() is True
        other.execute(text("SELECT pg_advisory_unlock(hashtextextended('eval:environment', 0))"))


def test_a_task_needs_one_full_payment_per_order() -> None:
    raw = TASKS["refund_eligible_001"].model_dump(mode="json")
    task_doc = {
        "id": "refund_eligible_001",
        "agent_input": {"customer_subject": "auth|alice", "user_message": "refund"},
        **{k: v for k, v in raw.items() if k not in ("task_id", "content_hash")},
    }
    task_doc["environment_fixture"]["payments"] = []
    evaluator = split_views(Task.model_validate(task_doc)).evaluator
    with pytest.raises(ValueError, match="needs one payment"):
        prepare(None, None, evaluator)  # type: ignore[arg-type]
