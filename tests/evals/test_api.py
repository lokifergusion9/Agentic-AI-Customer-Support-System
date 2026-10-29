"""The /eval API: operator-only, idempotent experiment creation, reads and comparisons (W5.4)."""

import io
import zipfile
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.auth.tokens import mint_token
from app.db.session import get_db
from evals import api, reports
from evals.execution import EvalContext
from tests.evals.test_experiment_execution import run_all
from tests.fakes import FakeEmbedder

KEY = "idem-1"
BODY = {
    "name": "api test",
    "dataset": "support-v1",
    "split": "pilot",
    "configs": ["scripted_correct", "scripted_false_success_claim"],
    "repetitions": 1,
}


@pytest.fixture
def session(conn: Connection) -> Session:
    return Session(bind=conn, join_transaction_mode="create_savepoint")


@pytest.fixture
def client(session, tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setattr(reports, "REPORTS_DIR", tmp_path)

    def test_db() -> Iterator[Session]:
        yield session

    api.app.dependency_overrides[get_db] = test_db
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def auth(role: str = "operator", subject: str = "auth|ops") -> dict[str, str]:
    return {"Authorization": f"Bearer {mint_token(subject, role)}"}


def create(client, body=None, key=KEY):
    return client.post(
        "/eval/experiments", json=body or BODY, headers={**auth(), "Idempotency-Key": key}
    )


@pytest.fixture
def ctx(mocks) -> EvalContext:
    return EvalContext(
        mocks=mocks, embedder=FakeEmbedder(), timeout_fault_seconds=0.01, sleep=lambda s: None
    )


@pytest.fixture
def finished(client, session, ctx) -> str:
    experiment_id = create(client).json()["experiment_id"]
    run_all(session, ctx)
    return experiment_id


# --- Access ------------------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["customer", "reviewer", "senior_reviewer"])
def test_only_an_operator_may_call_the_evaluation_api(client, role) -> None:
    assert client.get("/eval/datasets", headers=auth(role)).status_code == 403
    assert (
        client.post(
            "/eval/experiments", json=BODY, headers={**auth(role), "Idempotency-Key": "k"}
        ).status_code
        == 403
    )


def test_no_token_and_a_bad_token_are_unauthorized(client) -> None:
    assert client.get("/eval/datasets").status_code == 401
    assert client.get("/eval/datasets", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/eval/datasets", headers=auth()).status_code == 200


def test_an_operator_token_is_no_customer_token(client) -> None:
    from app.main import app as agent_app

    other = TestClient(agent_app)
    response = other.post("/tickets", json={"message": "hello there friend"}, headers=auth())
    assert response.status_code == 403


def test_every_route_is_under_the_eval_prefix() -> None:
    paths = api.app.openapi()["paths"]
    assert paths and all(path.startswith("/eval/") for path in paths)
    assert "/eval/runs/{run_id}" in paths  # the agent's own /runs/{run_id} is another application


# --- Datasets and configurations ---------------------------------------------------------------


def test_a_dataset_version_is_validated_imported_and_listed(client) -> None:
    imported = client.post("/eval/datasets/import", json={"dataset": "support-v1"}, headers=auth())
    assert imported.status_code == 200 and imported.json()["tasks"] == 50
    again = client.post("/eval/datasets/import", json={"dataset": "support-v1"}, headers=auth())
    assert again.json() == imported.json()
    listed = client.get("/eval/datasets", headers=auth()).json()
    assert [(d["id"], d["tasks"]) for d in listed] == [("support-v1", 50)]
    assert listed[0]["splits"] == {"dev": 30, "validation": 10, "test": 10, "pilot": 5}


@pytest.mark.parametrize("name", ["../etc", "support", "Support-v1", "support-v1/../x", ""])
def test_a_dataset_name_is_a_name_not_a_path(client, name) -> None:
    response = client.post("/eval/datasets/import", json={"dataset": name}, headers=auth())
    assert response.status_code == 422


def test_an_unknown_dataset_is_not_found(client) -> None:
    response = client.post("/eval/datasets/import", json={"dataset": "nothing-v9"}, headers=auth())
    assert response.status_code == 404


def test_configurations_are_registered_once_and_never_changed(client) -> None:
    first = client.post("/eval/agent-configs", json={"name": "lite_single_v1"}, headers=auth())
    again = client.post("/eval/agent-configs", json={"name": "lite_single_v1"}, headers=auth())
    assert (first.status_code, again.status_code) == (201, 200)
    assert first.json() == again.json() and first.json()["architecture"] == "single"

    custom = {
        "id": "mine",
        "model_id": "gemini-x",
        "prompt_version": "v1",
        "architecture": "single",
        "limits": {"max_steps": 10},
    }
    assert client.post("/eval/agent-configs", json=custom, headers=auth()).status_code == 201
    changed = {**custom, "limits": {"max_steps": 11}}
    conflict = client.post("/eval/agent-configs", json=changed, headers=auth())
    assert conflict.status_code == 409 and "immutable" in conflict.json()["detail"]

    listed = client.get("/eval/agent-configs", headers=auth()).json()
    assert {c["id"] for c in listed} == {"lite_single_v1", "mine"}


@pytest.mark.parametrize(
    "body",
    [
        {"name": "nope"},
        {"id": "x"},
        {
            "id": "x",
            "model_id": "m",
            "prompt_version": "v1",
            "architecture": "single",
            "limits": {"bogus": 1},
        },
    ],
)
def test_a_bad_configuration_is_rejected(client, body) -> None:
    assert client.post("/eval/agent-configs", json=body, headers=auth()).status_code == 422


# --- Creating experiments ----------------------------------------------------------------------


def test_creating_an_experiment_returns_its_id_status_and_planned_runs(client, session) -> None:
    response = create(client)
    assert response.status_code == 201
    body = response.json()
    assert (body["status"], body["planned_runs"]) == ("queued", 10)
    # The experiment, its runs and its jobs exist together.
    assert session.execute(text("SELECT count(*) FROM eval.runs")).scalar_one() == 10
    assert (
        session.execute(
            text("SELECT count(*) FROM support.jobs WHERE job_type = 'eval_attempt'")
        ).scalar_one()
        == 10
    )


def test_the_idempotency_key_returns_the_same_experiment(client, session) -> None:
    first, second = create(client), create(client)
    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json()["experiment_id"] == second.json()["experiment_id"]
    assert session.execute(text("SELECT count(*) FROM eval.experiments")).scalar_one() == 1


def test_reusing_a_key_for_a_different_request_is_rejected(client) -> None:
    create(client)
    other = create(client, {**BODY, "repetitions": 2})
    assert other.status_code == 409
    assert create(client, {**BODY, "configs": ["scripted_correct"]}).status_code == 409


def test_the_idempotency_key_is_required(client) -> None:
    assert client.post("/eval/experiments", json=BODY, headers=auth()).status_code == 422


@pytest.mark.parametrize(
    "change",
    [
        {"configs": ["nope"]},
        {"configs": ["scripted_correct", "scripted_correct"]},
        {"split": "nonexistent"},
        {"dataset": "nothing-v9"},
        {"repetitions": 0},
        {"extra": 1},
    ],
)
def test_a_bad_experiment_request_is_rejected(client, change) -> None:
    assert create(client, {**BODY, **change}, key="bad").status_code in (404, 422)


def test_the_held_out_test_split_needs_an_explicit_acknowledgement(client) -> None:
    assert create(client, {**BODY, "split": "test"}, key="t1").status_code == 422
    allowed = create(client, {**BODY, "split": "test", "allow_test_split": True}, key="t2")
    assert allowed.status_code == 201 and allowed.json()["planned_runs"] == 20


# --- Reading results ---------------------------------------------------------------------------


def test_the_experiment_shows_progress_and_the_scorecard(client, finished) -> None:
    detail = client.get(f"/eval/experiments/{finished}", headers=auth()).json()
    assert (detail["status"], detail["planned_runs"]) == ("completed", 10)
    assert detail["progress"] == {"completed": 10}
    good = detail["configs"]["scripted_correct"]["verified_success_rate"]
    assert good == {"value": 1.0, "n": 5, "of": 5}
    assert detail["scorecards"]["scripted_false_success_claim"]["overall"]["scorable_runs"] == 5
    assert detail["manifest"]["repetitions"] == 1
    listed = client.get("/eval/experiments", headers=auth()).json()
    assert [e["id"] for e in listed] == [finished]


def test_an_unknown_experiment_run_or_attempt_is_not_found(client) -> None:
    for path in (
        "/eval/experiments/x",
        "/eval/experiments/x/runs",
        "/eval/runs/x",
        "/eval/attempts/x/events",
        "/eval/experiments/x/export",
    ):
        assert client.get(path, headers=auth()).status_code == 404, path


def test_runs_are_filtered_by_category_status_and_reason_code(client, finished) -> None:
    def runs(**params):
        return client.get(
            f"/eval/experiments/{finished}/runs", params=params, headers=auth()
        ).json()

    assert runs()["total"] == 10
    assert {r["category"] for r in runs(category="eligible_refund")["runs"]} == {"eligible_refund"}
    assert runs(category="eligible_refund")["total"] == 2
    failed = runs(status="fail")
    assert failed["total"] == 2 and {r["config_id"] for r in failed["runs"]} == {
        "scripted_false_success_claim"
    }
    by_reason = runs(reason_code="FALSE_REFUND_CLAIM")
    assert {r["task_id"] for r in by_reason["runs"]} == {
        "refund_eligible_001",
        "ineligible_return_in_transit_001",
    }
    assert runs(status="completed")["total"] == 10  # execution status matches too
    assert runs(config_id="scripted_correct", task_id="lookup_delivered_001")["total"] == 1


def test_runs_are_paginated(client, finished) -> None:
    page = client.get(
        f"/eval/experiments/{finished}/runs", params={"limit": 4, "offset": 8}, headers=auth()
    ).json()
    assert (page["total"], len(page["runs"]), page["limit"], page["offset"]) == (10, 2, 4, 8)
    assert (
        client.get(
            f"/eval/experiments/{finished}/runs", params={"limit": 500}, headers=auth()
        ).status_code
        == 422
    )


def failing_run(client, experiment_id) -> dict:
    runs = client.get(
        f"/eval/experiments/{experiment_id}/runs",
        params={"reason_code": "FALSE_REFUND_CLAIM", "task_id": "refund_eligible_001"},
        headers=auth(),
    ).json()["runs"]
    return client.get(f"/eval/runs/{runs[0]['id']}", headers=auth()).json()


def test_a_run_shows_what_the_agent_saw_apart_from_the_ground_truth(client, finished) -> None:
    run = failing_run(client, finished)

    assert set(run["agent_view"]) == {"task_id", "agent_input"}
    assert "evaluation" not in run["agent_view"] and "fixture" not in str(run["agent_view"]).lower()
    assert run["ground_truth"]["outcome"]["refund_total_minor"] == 249900
    assert (run["execution_status"], run["evaluation_status"]) == ("completed", "fail")
    assert len(run["attempts"]) == 1 and run["final_attempt_id"] == run["attempts"][0]["id"]
    failed = {
        (s["evaluator_name"], s["dimension"], s["reason_code"])
        for s in run["scores"]
        if s["status"] == "fail"
    }
    assert ("response", "refund_claim", "FALSE_REFUND_CLAIM") in failed
    assert ("outcome", "amount", "MISSING_REFUND") in failed


def test_a_run_lists_the_state_changes_against_what_was_expected(client, finished) -> None:
    changes = {c["what"]: c for c in failing_run(client, finished)["state_changes"]}
    refunds = changes["refunds at the payment provider"]
    assert (refunds["expected"], refunds["observed"], refunds["matches"]) == (1, 0, False)
    assert changes["ticket status"]["matches"] is True


def test_the_trace_is_ordered_and_holds_structured_events(client, finished) -> None:
    run = failing_run(client, finished)
    events = client.get(f"/eval/attempts/{run['final_attempt_id']}/events", headers=auth()).json()
    sequence = [e["sequence_number"] for e in events["events"]]
    assert sequence == sorted(sequence) and len(sequence) > 5
    assert {"tool_requested", "agent_finished"} <= {e["event_type"] for e in events["events"]}
    assert "reasoning" not in str(events).lower().replace("reasoning_tokens", "")


def test_two_configurations_are_compared(client, finished) -> None:
    body = {
        "baseline": {"experiment_id": finished, "config_id": "scripted_correct"},
        "candidate": {"experiment_id": finished, "config_id": "scripted_false_success_claim"},
    }
    result = client.post("/eval/comparisons", json=body, headers=auth())
    assert result.status_code == 200
    assert result.json()["regressed_tasks"] == [
        "ineligible_return_in_transit_001",
        "refund_eligible_001",
    ]
    assert (
        client.post(
            "/eval/comparisons",
            json={**body, "candidate": {"experiment_id": finished}},
            headers=auth(),
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/eval/comparisons",
            json={"baseline": {"experiment_id": "x"}, "candidate": {"experiment_id": "y"}},
            headers=auth(),
        ).status_code
        == 404
    )


def test_incompatible_experiments_are_not_compared(client, finished) -> None:
    other = create(
        client, {**BODY, "split": "validation", "configs": ["scripted_correct"]}, "v"
    ).json()
    body = {
        "baseline": {"experiment_id": finished, "config_id": "scripted_correct"},
        "candidate": {"experiment_id": other["experiment_id"]},
    }
    response = client.post("/eval/comparisons", json=body, headers=auth())
    assert response.status_code == 409
    assert any("split" in d for d in response.json()["detail"]["differences"])


def test_the_report_downloads_as_a_zip(client, finished) -> None:
    response = client.get(f"/eval/experiments/{finished}/export", headers=auth())
    assert response.status_code == 200 and response.headers["content-type"] == "application/zip"
    names = sorted(zipfile.ZipFile(io.BytesIO(response.content)).namelist())
    assert names == [
        f"{finished}/{n}"
        for n in ("manifest.json", "runs.csv", "runs.json", "scorecard.csv", "scorecard.json")
    ]


# --- Cancelling --------------------------------------------------------------------------------


def test_cancelling_stops_the_queued_runs(client, session, ctx) -> None:
    experiment_id = create(client).json()["experiment_id"]
    response = client.post(f"/eval/experiments/{experiment_id}/cancel", headers=auth())
    assert response.json() == {
        "experiment_id": experiment_id,
        "status": "cancelled",
        "runs_cancelled": 10,
    }
    run_all(session, ctx)  # the jobs still exist; they find the runs already cancelled
    assert session.execute(text("SELECT count(*) FROM eval.attempts")).scalar_one() == 0
    assert (
        client.post(f"/eval/experiments/{experiment_id}/cancel", headers=auth()).status_code == 409
    )


def test_a_finished_experiment_cannot_be_cancelled(client, finished) -> None:
    assert client.post(f"/eval/experiments/{finished}/cancel", headers=auth()).status_code == 409
