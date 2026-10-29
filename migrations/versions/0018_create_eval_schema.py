"""Create the `eval` schema for the evaluation platform (W4.3, E§16, Q8).

One migration for the whole schema: it is a separate unit from the `support` tables and is
upgraded and dropped as one. The agent's tools read only `support`, so expected outcomes stored
here are out of the agent's reach.

Definitions are immutable once written (dataset versions, task versions, prompt versions,
agent configs): a trigger refuses UPDATE, so a stored score always refers to exactly what ran.
An experiment's manifest is immutable too; only its status and timestamps move.

Attempts keep `workflow_run_id` without a foreign key: the `support` tables are reset before
every attempt, which is why the trace and the final state are copied into `eval` (W4.4).

Revision ID: 0018
Revises: 0017
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from evals.db import EVAL_SCHEMA as S

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

IMMUTABLE_TABLES = ("dataset_versions", "task_versions", "prompt_versions", "agent_configs")


def created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )


def fk(
    table: str, column: str, target: str, *, nullable: bool = False, ondelete: str | None = None
):
    return sa.Column(
        column,
        sa.Text(),
        sa.ForeignKey(f"{S}.{target}", name=f"fk_{table}_{column}", ondelete=ondelete),
        nullable=nullable,
    )


def jsonb(name: str, *, nullable: bool = False, default: str | None = "'{}'::jsonb") -> sa.Column:
    server_default = sa.text(default) if default and not nullable else None
    return sa.Column(name, postgresql.JSONB(), nullable=nullable, server_default=server_default)


def upgrade() -> None:
    op.execute(f'CREATE SCHEMA "{S}"')

    op.create_table(
        "datasets",
        sa.Column("id", sa.Text(), primary_key=True),  # e.g. support
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        created_at(),
        schema=S,
    )
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.Text(), primary_key=True),  # e.g. support-v1
        fk("dataset_versions", "dataset_id", "datasets.id"),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        jsonb("manifest"),  # task ID -> {version, content_hash}
        jsonb("splits"),  # split name -> [task IDs]
        created_at(),
        sa.UniqueConstraint("dataset_id", "version", name="uq_dataset_versions_version"),
        sa.CheckConstraint("version > 0", name="ck_dataset_versions_version"),
        schema=S,
    )
    op.create_table(
        "task_versions",
        sa.Column("id", sa.Text(), primary_key=True),  # e.g. refund_eligible_001@v1
        sa.Column("task_id", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("difficulty", sa.Text(), nullable=False),
        sa.Column("scenario_family", sa.Text(), nullable=False),
        jsonb("agent_view", default=None),
        jsonb("evaluator_view", default=None),
        created_at(),
        sa.UniqueConstraint("task_id", "version", name="uq_task_versions_version"),
        sa.CheckConstraint("version > 0", name="ck_task_versions_version"),
        schema=S,
    )
    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.Text(), primary_key=True),  # e.g. resolution_v1
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        created_at(),
        sa.UniqueConstraint("name", "version", name="uq_prompt_versions_version"),
        schema=S,
    )
    op.create_table(
        "agent_configs",
        sa.Column("id", sa.Text(), primary_key=True),  # e.g. flashlite_specialist_v1
        sa.Column("adapter", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("architecture", sa.Text(), nullable=False),
        jsonb("prompt_hashes"),  # prompt name -> sha256, for every prompt the config uses
        jsonb("limits"),  # max_steps, max_tokens, temperature, max_output_tokens, retries
        sa.Column("tool_schema_hash", sa.Text(), nullable=False),
        # Hash of everything above but the id: the same settings never get two names.
        sa.Column("config_hash", sa.Text(), nullable=False),
        created_at(),
        sa.UniqueConstraint("config_hash", name="uq_agent_configs_config_hash"),
        sa.CheckConstraint("adapter IN ('langgraph', 'scripted')", name="ck_agent_configs_adapter"),
        sa.CheckConstraint(
            "architecture IN ('single', 'specialist', 'scripted')",
            name="ck_agent_configs_architecture",
        ),
        schema=S,
    )
    op.create_table(
        "experiments",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        # A retried create returns the existing experiment instead of a second one (E§18).
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        fk("experiments", "dataset_version_id", "dataset_versions.id"),
        sa.Column("split", sa.Text(), nullable=False),
        sa.Column("repetitions", sa.Integer(), nullable=False),
        # Configs, limits, environment and evaluator versions, code commit, lockfile hash.
        jsonb("manifest", default=None),
        sa.Column("status", sa.Text(), nullable=False, server_default="queued"),
        created_at(),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("idempotency_key", name="uq_experiments_idempotency_key"),
        sa.CheckConstraint("repetitions > 0", name="ck_experiments_repetitions"),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_experiments_status",
        ),
        schema=S,
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.Text(), primary_key=True),
        fk("runs", "experiment_id", "experiments.id"),
        fk("runs", "agent_config_id", "agent_configs.id"),
        fk("runs", "task_version_id", "task_versions.id"),
        sa.Column("repetition", sa.Integer(), nullable=False),
        sa.Column("execution_status", sa.Text(), nullable=False, server_default="queued"),
        sa.Column("evaluation_status", sa.Text(), nullable=False, server_default="pending"),
        created_at(),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "experiment_id",
            "agent_config_id",
            "task_version_id",
            "repetition",
            name="uq_runs_config_task_repetition",
        ),
        sa.CheckConstraint("repetition > 0", name="ck_runs_repetition"),
        sa.CheckConstraint(
            "execution_status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_runs_execution_status",
        ),
        sa.CheckConstraint(
            "evaluation_status IN ('pending', 'scored', 'failed', 'skipped')",
            name="ck_runs_evaluation_status",
        ),
        schema=S,
    )
    op.create_index(
        "ix_runs_experiment_status", "runs", ["experiment_id", "execution_status"], schema=S
    )

    op.create_table(
        "attempts",
        sa.Column("id", sa.Text(), primary_key=True),
        fk("attempts", "run_id", "runs.id"),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="running"),
        sa.Column("environment_id", sa.Text(), nullable=False),
        sa.Column("workflow_run_id", sa.Text(), nullable=True),  # support.workflow_runs, reset
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        # Monotonic durations; waiting for the (simulated) reviewer is not active time (W4.5).
        sa.Column("active_ms", sa.BigInteger(), nullable=True),
        sa.Column("approval_wait_ms", sa.BigInteger(), nullable=True),
        sa.Column("stop_reason", sa.Text(), nullable=True),
        sa.Column("final_response", sa.Text(), nullable=True),
        jsonb("usage"),  # summed model-call tokens; null fields stay null (unknown, W2.16)
        jsonb("error", nullable=True),
        sa.UniqueConstraint("run_id", "attempt_number", name="uq_attempts_run_attempt_number"),
        sa.CheckConstraint("attempt_number > 0", name="ck_attempts_attempt_number"),
        sa.CheckConstraint("status IN ('running', 'finished', 'error')", name="ck_attempts_status"),
        sa.CheckConstraint(
            "(status = 'running') = (finished_at IS NULL)", name="ck_attempts_finished_at"
        ),
        schema=S,
    )
    op.create_table(
        "trace_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        fk("trace_events", "attempt_id", "attempts.id", ondelete="CASCADE"),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("source_event_id", sa.BigInteger(), nullable=True),  # support.audit_events.id
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        jsonb("payload"),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        # Also the index for reading an attempt's events in order.
        sa.UniqueConstraint("attempt_id", "sequence_number", name="uq_trace_events_sequence"),
        schema=S,
    )
    op.create_table(
        "artifacts",
        sa.Column("id", sa.Text(), primary_key=True),
        fk("artifacts", "attempt_id", "attempts.id", nullable=True, ondelete="CASCADE"),
        fk("artifacts", "experiment_id", "experiments.id", nullable=True),
        sa.Column("kind", sa.Text(), nullable=False),
        jsonb("content", nullable=True),  # the final-state snapshot, stored inline
        sa.Column("path", sa.Text(), nullable=True),  # an exported report on disk
        sa.Column("content_hash", sa.Text(), nullable=False),
        created_at(),
        sa.CheckConstraint(
            "kind IN ('final_state', 'report_json', 'report_csv')", name="ck_artifacts_kind"
        ),
        sa.CheckConstraint(
            "(kind = 'final_state') = (attempt_id IS NOT NULL) "
            "AND (kind = 'final_state') = (experiment_id IS NULL)",
            name="ck_artifacts_owner",
        ),
        sa.CheckConstraint(
            "content IS NOT NULL OR path IS NOT NULL", name="ck_artifacts_content_or_path"
        ),
        schema=S,
    )
    op.create_index(
        "uq_artifacts_attempt_kind",
        "artifacts",
        ["attempt_id", "kind"],
        unique=True,
        schema=S,
        postgresql_where=sa.text("attempt_id IS NOT NULL"),
    )
    op.create_table(
        "scores",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        fk("scores", "attempt_id", "attempts.id", ondelete="CASCADE"),
        # Copied from the run and the task so scorecards filter without joins (W5.1).
        fk("scores", "experiment_id", "experiments.id"),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("evaluator_name", sa.Text(), nullable=False),
        sa.Column("evaluator_version", sa.Text(), nullable=False),
        sa.Column("dimension", sa.Text(), nullable=False),
        # The score contract (W4.9): not_applicable and error are not failures.
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("score", sa.Numeric(5, 4), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        jsonb("details"),  # evidence: event sequence numbers, snapshot fields
        created_at(),
        sa.UniqueConstraint(
            "attempt_id",
            "evaluator_name",
            "evaluator_version",
            "dimension",
            name="uq_scores_attempt_dimension",
        ),
        sa.CheckConstraint(
            "status IN ('scored', 'not_applicable', 'error')", name="ck_scores_status"
        ),
        sa.CheckConstraint("(status = 'scored') = (passed IS NOT NULL)", name="ck_scores_passed"),
        sa.CheckConstraint("score IS NULL OR score BETWEEN 0 AND 1", name="ck_scores_range"),
        schema=S,
    )
    op.create_index(
        "ix_scores_experiment_category", "scores", ["experiment_id", "category"], schema=S
    )

    op.create_table(
        "price_versions",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False),
        # USD per million tokens at paid rates; the free tier itself costs $0 (Q4).
        sa.Column("input_per_mtok_usd", sa.Numeric(10, 4), nullable=False),
        sa.Column("output_per_mtok_usd", sa.Numeric(10, 4), nullable=False),
        sa.Column("cached_input_per_mtok_usd", sa.Numeric(10, 4), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),  # where the rate was read, and when
        created_at(),
        sa.UniqueConstraint(
            "provider", "model_id", "effective_from", name="uq_price_versions_effective"
        ),
        sa.CheckConstraint(
            "input_per_mtok_usd >= 0 AND output_per_mtok_usd >= 0 "
            "AND (cached_input_per_mtok_usd IS NULL OR cached_input_per_mtok_usd >= 0)",
            name="ck_price_versions_rates",
        ),
        schema=S,
    )

    op.execute(
        f"""
        CREATE FUNCTION {S}.refuse_update() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION '%.% is immutable; register a new version instead',
                TG_TABLE_SCHEMA, TG_TABLE_NAME;
        END
        $$
        """
    )
    for table in IMMUTABLE_TABLES:
        op.execute(
            f"CREATE TRIGGER trg_{table}_immutable BEFORE UPDATE ON {S}.{table} "
            f"FOR EACH ROW EXECUTE FUNCTION {S}.refuse_update()"
        )
    op.execute(
        f"""
        CREATE FUNCTION {S}.keep_experiment_manifest() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF (NEW.id, NEW.name, NEW.idempotency_key, NEW.dataset_version_id, NEW.split,
                NEW.repetitions, NEW.manifest, NEW.created_at)
               IS DISTINCT FROM
               (OLD.id, OLD.name, OLD.idempotency_key, OLD.dataset_version_id, OLD.split,
                OLD.repetitions, OLD.manifest, OLD.created_at) THEN
                RAISE EXCEPTION 'an experiment manifest is immutable; only status and times change';
            END IF;
            RETURN NEW;
        END
        $$
        """
    )
    op.execute(
        f"CREATE TRIGGER trg_experiments_manifest BEFORE UPDATE ON {S}.experiments "
        f"FOR EACH ROW EXECUTE FUNCTION {S}.keep_experiment_manifest()"
    )


def downgrade() -> None:
    op.execute(f'DROP SCHEMA "{S}" CASCADE')
