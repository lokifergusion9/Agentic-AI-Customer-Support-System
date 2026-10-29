"""The `eval` schema: tables, constraints, indexes and immutability (W4.3, E§16, Q8)."""

import pytest
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import IntegrityError, ProgrammingError

from evals.db import EVAL_SCHEMA as S

EVAL_TABLES = {
    "datasets",
    "dataset_versions",
    "task_versions",
    "prompt_versions",
    "agent_configs",
    "experiments",
    "runs",
    "attempts",
    "trace_events",
    "artifacts",
    "scores",
    "price_versions",
}


def insert(conn: Connection, table: str, **values: object) -> None:
    columns = ", ".join(values)
    params = ", ".join(
        f"CAST(:{k} AS jsonb)" if isinstance(v, str) and v.startswith(("{", "[")) else f":{k}"
        for k, v in values.items()
    )
    conn.execute(text(f"INSERT INTO {S}.{table} ({columns}) VALUES ({params})"), values)


@pytest.fixture
def run(conn: Connection) -> Connection:
    """One experiment with one run and one attempt, rolled back after the test."""
    insert(conn, "datasets", id="support")
    insert(
        conn,
        "dataset_versions",
        id="support-v1",
        dataset_id="support",
        version=1,
        content_hash="sha256:d",
    )
    insert(
        conn,
        "task_versions",
        id="refund_eligible_001@v1",
        task_id="refund_eligible_001",
        version=1,
        content_hash="sha256:t",
        category="eligible_refund",
        difficulty="medium",
        scenario_family="headphones_return_received",
        agent_view='{"task_id": "refund_eligible_001"}',
        evaluator_view='{"task_id": "refund_eligible_001"}',
    )
    insert(
        conn,
        "agent_configs",
        id="flashlite_specialist_v1",
        adapter="langgraph",
        provider="gemini",
        model_id="gemini-3.5-flash-lite",
        prompt_version="v1",
        architecture="specialist",
        tool_schema_hash="sha256:tools",
        config_hash="sha256:c1",
    )
    insert(
        conn,
        "experiments",
        id="exp_1",
        name="pilot",
        idempotency_key="pilot-1",
        dataset_version_id="support-v1",
        split="pilot",
        repetitions=1,
        manifest='{"configs": ["flashlite_specialist_v1"]}',
    )
    insert(
        conn,
        "runs",
        id="erun_1",
        experiment_id="exp_1",
        agent_config_id="flashlite_specialist_v1",
        task_version_id="refund_eligible_001@v1",
        repetition=1,
    )
    insert(conn, "attempts", id="att_1", run_id="erun_1", attempt_number=1, environment_id="env_1")
    return conn


def rejects(conn: Connection, sql: str, error: type[Exception] = IntegrityError, **params) -> str:
    savepoint = conn.begin_nested()
    with pytest.raises(error) as caught:
        conn.execute(text(sql), params)
    savepoint.rollback()
    return str(caught.value)


def test_eval_tables_live_in_their_own_schema(engine: Engine) -> None:
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema = :s"),
            {"s": S},
        )
        assert {row.table_name for row in rows} == EVAL_TABLES


def test_one_run_per_config_task_and_repetition(run: Connection) -> None:
    sql = (
        f"INSERT INTO {S}.runs (id, experiment_id, agent_config_id, task_version_id, repetition) "
        "VALUES ('erun_2', 'exp_1', 'flashlite_specialist_v1', 'refund_eligible_001@v1', :rep)"
    )
    assert "uq_runs_config_task_repetition" in rejects(run, sql, rep=1)
    run.execute(text(sql), {"rep": 2})  # another repetition is a new run


def test_attempt_numbers_are_unique_per_run(run: Connection) -> None:
    sql = (
        f"INSERT INTO {S}.attempts (id, run_id, attempt_number, environment_id) "
        "VALUES ('att_2', 'erun_1', 1, 'env_2')"
    )
    assert "uq_attempts_run_attempt_number" in rejects(run, sql)


def test_trace_sequence_numbers_are_unique_per_attempt(run: Connection) -> None:
    sql = (
        f"INSERT INTO {S}.trace_events (attempt_id, sequence_number, actor, event_type, "
        "occurred_at) VALUES ('att_1', 1, 'system:workflow', 'run_started', now())"
    )
    run.execute(text(sql))
    assert "uq_trace_events_sequence" in rejects(run, sql)


def test_one_score_per_evaluator_version_and_dimension(run: Connection) -> None:
    sql = (
        f"INSERT INTO {S}.scores (attempt_id, experiment_id, category, evaluator_name, "
        "evaluator_version, dimension, status, score, reason_code) VALUES ('att_1', 'exp_1', "
        "'eligible_refund', 'outcome', :version, 'amount', 'pass', 1, 'OK')"
    )
    run.execute(text(sql), {"version": "1"})
    assert "uq_scores_attempt_dimension" in rejects(run, sql, version="1")
    run.execute(text(sql), {"version": "2"})  # a new evaluator version scores again


def test_score_contract_keeps_not_applicable_apart_from_failure(run: Connection) -> None:
    """W4.9: not_applicable is exactly a score that does not apply, never a hidden fail."""
    sql = (
        f"INSERT INTO {S}.scores (attempt_id, experiment_id, category, evaluator_name, "
        "evaluator_version, dimension, applicable, status) VALUES ('att_1', 'exp_1', "
        "'eligible_refund', 'policy', '1', :dimension, :applicable, :status)"
    )
    assert "ck_scores_applicable" in rejects(
        run, sql, dimension="a", applicable=True, status="not_applicable"
    )
    assert "ck_scores_applicable" in rejects(
        run, sql, dimension="b", applicable=False, status="fail"
    )
    assert "ck_scores_status" in rejects(run, sql, dimension="c", applicable=True, status="scored")
    run.execute(text(sql), {"dimension": "d", "applicable": False, "status": "not_applicable"})
    run.execute(text(sql), {"dimension": "e", "applicable": True, "status": "evaluation_error"})


def test_execution_and_evaluation_status_are_separate(run: Connection) -> None:
    """W4.8: a confident wrong answer is completed + fail."""
    run.execute(
        text(
            f"UPDATE {S}.runs SET execution_status = 'completed', evaluation_status = 'fail' "
            "WHERE id = 'erun_1'"
        )
    )
    bad = f"UPDATE {S}.runs SET execution_status = :s WHERE id = 'erun_1'"
    assert "ck_runs_execution_status" in rejects(run, bad, s="succeeded")
    run.execute(text(bad), {"s": "budget_exceeded"})


@pytest.mark.parametrize(
    ("table", "key"),
    [
        ("agent_configs", "flashlite_specialist_v1"),
        ("task_versions", "refund_eligible_001@v1"),
        ("dataset_versions", "support-v1"),
    ],
)
def test_definitions_are_immutable(run: Connection, table: str, key: str) -> None:
    message = rejects(
        run, f"UPDATE {S}.{table} SET created_at = now() WHERE id = :id", ProgrammingError, id=key
    )
    assert "immutable" in message


def test_experiment_manifest_is_immutable_but_status_moves(run: Connection) -> None:
    message = rejects(
        run,
        f"UPDATE {S}.experiments SET manifest = '{{}}'::jsonb WHERE id = 'exp_1'",
        ProgrammingError,
    )
    assert "immutable" in message
    run.execute(
        text(
            f"UPDATE {S}.experiments SET status = 'running', started_at = now() WHERE id = 'exp_1'"
        )
    )


def test_experiment_idempotency_key_is_unique(run: Connection) -> None:
    sql = (
        f"INSERT INTO {S}.experiments (id, name, idempotency_key, dataset_version_id, split, "
        "repetitions, manifest) VALUES ('exp_2', 'again', 'pilot-1', 'support-v1', 'pilot', 1, "
        "'{}'::jsonb)"
    )
    assert "uq_experiments_idempotency_key" in rejects(run, sql)


def test_same_settings_never_get_two_config_names(run: Connection) -> None:
    sql = (
        f"INSERT INTO {S}.agent_configs (id, adapter, provider, model_id, prompt_version, "
        "architecture, tool_schema_hash, config_hash) VALUES ('renamed', 'langgraph', 'gemini', "
        "'gemini-3.5-flash-lite', 'v1', 'specialist', 'sha256:tools', 'sha256:c1')"
    )
    assert "uq_agent_configs_config_hash" in rejects(run, sql)


def test_final_state_artifact_belongs_to_one_attempt(run: Connection) -> None:
    sql = (
        f"INSERT INTO {S}.artifacts (id, attempt_id, experiment_id, kind, content, content_hash) "
        "VALUES (:id, :attempt, :experiment, :kind, '{}'::jsonb, 'sha256:a')"
    )
    run.execute(
        text(sql), {"id": "art_1", "attempt": "att_1", "experiment": None, "kind": "final_state"}
    )
    assert "uq_artifacts_attempt_kind" in rejects(
        run, sql, id="art_2", attempt="att_1", experiment=None, kind="final_state"
    )
    assert "ck_artifacts_owner" in rejects(
        run, sql, id="art_3", attempt=None, experiment="exp_1", kind="final_state"
    )
    run.execute(
        text(sql), {"id": "art_4", "attempt": None, "experiment": "exp_1", "kind": "report_json"}
    )


def test_filtered_columns_are_indexed(engine: Engine) -> None:
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = :s"), {"s": S}
        )
        indexes = {row.indexname: row.indexdef for row in rows}
    assert "(experiment_id, execution_status)" in indexes["ix_runs_experiment_status"]
    assert "(attempt_id, sequence_number)" in indexes["uq_trace_events_sequence"]
    assert "(experiment_id, category)" in indexes["ix_scores_experiment_category"]


def test_flexible_payloads_are_jsonb(engine: Engine) -> None:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = :s AND data_type = 'jsonb'"
            ),
            {"s": S},
        )
        jsonb = {(row.table_name, row.column_name) for row in rows}
    assert {
        ("task_versions", "agent_view"),
        ("task_versions", "evaluator_view"),
        ("experiments", "manifest"),
        ("trace_events", "payload"),
        ("artifacts", "content"),
        ("scores", "evidence"),
    } <= jsonb
