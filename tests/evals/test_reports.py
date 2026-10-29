"""Report export and rerunning an experiment from its saved manifest (W5.5)."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from evals import metrics
from evals.comparison import Side, compare
from evals.configs import scripted_config_row
from evals.execution import EvalContext
from evals.experiments import ExperimentPlan, create_experiment, plan_from_experiment
from evals.reports import export_experiment
from tests.evals.test_experiment_execution import run_all
from tests.fakes import FakeEmbedder


@pytest.fixture
def session(conn: Connection) -> Session:
    return Session(bind=conn, join_transaction_mode="create_savepoint")


@pytest.fixture
def ctx(mocks) -> EvalContext:
    return EvalContext(
        mocks=mocks, embedder=FakeEmbedder(), timeout_fault_seconds=0.01, sleep=lambda s: None
    )


@pytest.fixture
def finished(session, ctx) -> str:
    plan = ExperimentPlan(
        "report",
        "support-v1",
        "pilot",
        [scripted_config_row("correct"), scripted_config_row("false_success_claim")],
        1,
        "report",
    )
    experiment_id, _ = create_experiment(session, plan)
    run_all(session, ctx)
    return experiment_id


def read(directory, name):
    return (directory / name).read_text(encoding="utf-8")


def test_the_report_has_the_manifest_scorecard_and_one_row_per_run(session, finished, tmp_path):
    directory = export_experiment(session.connection(), finished, tmp_path)

    assert sorted(p.name for p in directory.iterdir()) == [
        "manifest.json",
        "runs.csv",
        "runs.json",
        "scorecard.csv",
        "scorecard.json",
    ]
    rows = list(csv.DictReader(io.StringIO(read(directory, "runs.csv"))))
    assert len(rows) == 10 == len(json.loads(read(directory, "runs.json")))
    refund = next(
        r
        for r in rows
        if r["task_id"] == "refund_eligible_001"
        and r["config_id"] == "scripted_false_success_claim"
    )
    assert refund["verified_success"] == "False"
    assert "response.refund_claim:FALSE_REFUND_CLAIM" in refund["failures"]
    assert refund["human_assisted"] == "False"  # it never asked for an approval


def test_the_manifest_records_what_is_needed_to_reproduce_the_run(session, finished, tmp_path):
    manifest = json.loads(
        read(export_experiment(session.connection(), finished, tmp_path), "manifest.json")
    )

    experiment = manifest["experiment"]
    assert (experiment["dataset_version"], experiment["split"]) == ("support-v1", "pilot")
    assert experiment["dataset_hash"].startswith("sha256:")
    inner = manifest["manifest"]
    assert inner["lockfile_hash"].startswith("sha256:") and inner["code_commit"]
    assert inner["environment_version"] == "1"
    assert set(inner["evaluator_versions"]) == {"outcome", "policy", "trajectory", "response"}
    config = manifest["configs"]["scripted_correct"]
    assert config["tool_schema_hash"].startswith("sha256:") and config["adapter"] == "scripted"
    assert "limits" in inner and len(inner["tasks"]) == 5
    # The fault configuration of the pilot's tool-failure task is part of what ran.
    faults = manifest["fault_configuration"]["fault_shipping_transient_001"]
    assert faults[0]["operation"] == "shipping.get_shipment" and faults[0]["times"] == 2


def test_model_ids_are_the_ones_the_provider_reported(session, finished, tmp_path) -> None:
    connection = session.connection()
    connection.execute(
        text(
            "INSERT INTO eval.trace_events (attempt_id, sequence_number, actor, event_type, "
            "payload, occurred_at) SELECT a.id, 9999, 'agent:x', 'model_call', "
            '\'{"outcome": "ok", "resolved_model": "gemini-3.5-flash-lite-001"}\'::jsonb, now() '
            "FROM eval.attempts a JOIN eval.runs r ON r.id = a.run_id "
            "WHERE r.agent_config_id = 'scripted_correct' LIMIT 1"
        )
    )
    manifest = json.loads(read(export_experiment(connection, finished, tmp_path), "manifest.json"))
    assert manifest["configs"]["scripted_correct"]["resolved_model_ids"] == [
        "gemini-3.5-flash-lite-001"
    ]
    assert "resolved_model_ids" not in manifest["configs"]["scripted_false_success_claim"]


def test_the_scorecard_files_match_the_metrics(session, finished, tmp_path) -> None:
    directory = export_experiment(session.connection(), finished, tmp_path)
    cards = json.loads(read(directory, "scorecard.json"))
    live = metrics.experiment_scorecard(session.connection(), finished)
    assert (
        cards["scripted_correct"]["overall"]["verified_success_rate"]
        == (live["scripted_correct"]["overall"]["verified_success_rate"])
    )
    table = list(csv.DictReader(io.StringIO(read(directory, "scorecard.csv"))))
    overall = {r["config_id"]: r for r in table if r["category"] == "ALL"}
    assert float(overall["scripted_correct"]["verified_success_rate"]) == 1.0
    assert float(overall["scripted_false_success_claim"]["verified_success_rate"]) == 0.6
    assert {
        "verified_success_rate",
        "evaluation_coverage",
        "executed_violation_rate",
        "estimated_cost_usd",
    } <= set(table[0])


def test_exporting_again_gives_the_same_bytes_and_replaces_the_artifacts(
    session, finished, tmp_path
):
    connection = session.connection()
    first = export_experiment(connection, finished, tmp_path)
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    second = export_experiment(connection, finished, tmp_path)

    assert {p.name: p.read_bytes() for p in second.iterdir()} == before
    artifacts = connection.execute(
        text("SELECT path, content_hash, kind FROM eval.artifacts WHERE experiment_id = :id"),
        {"id": finished},
    ).all()
    assert len(artifacts) == 5
    assert {a.kind for a in artifacts} == {"report_json", "report_csv"}
    for artifact in artifacts:
        digest = hashlib.sha256(Path(artifact.path).read_bytes()).hexdigest()
        assert artifact.content_hash == f"sha256:{digest}"


def test_a_saved_experiment_reruns_from_its_manifest_and_compares_clean(session, ctx, finished):
    connection = session.connection()
    plan = plan_from_experiment(connection, finished, idempotency_key="rerun-1")
    assert (plan.dataset, plan.split, plan.repetitions) == ("support-v1", "pilot", 1)
    assert sorted(row["id"] for row in plan.configs) == [
        "scripted_correct",
        "scripted_false_success_claim",
    ]

    again, created = create_experiment(session, plan)
    run_all(session, ctx)

    assert created and again != finished
    for config in ("scripted_correct", "scripted_false_success_claim"):
        result = compare(connection, Side(finished, config), Side(again, config))
        assert result["compatible"] and result["tasks_compared"] == 5
        assert result["regressed_tasks"] == [] and result["improved_tasks"] == []
        assert result["differences"]["verified_success_rate"]["difference"] == 0
