"""Create jobs: the PostgreSQL work queue for the worker process (W3.13, Q7).

One row per unit of background work. `job_type` picks the handler (`workflow_run` now,
`eval_attempt` in W4.8). Workers claim with `FOR UPDATE SKIP LOCKED` and hold a lease: every
write that finishes a job must present the current `lease_token`, so a worker whose lease
expired and was reclaimed can never finalize the job. `dedupe_key` (e.g. `run:<run_id>`)
allows at most one queued or running job per key, so two workers never drive the same run.

Revision ID: 0015
Revises: 0014
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.db import SCHEMA

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATUSES = "('queued', 'running', 'completed', 'failed', 'cancelled')"


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("job_type", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="queued"),
        sa.Column("dedupe_key", sa.Text(), nullable=True),
        sa.Column(
            "available_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("worker_id", sa.Text(), nullable=True),
        sa.Column("lease_token", sa.Text(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"status IN {STATUSES}", name="ck_jobs_status"),
        sa.CheckConstraint("job_type IN ('workflow_run', 'eval_attempt')", name="ck_jobs_job_type"),
        sa.CheckConstraint("attempts >= 0 AND max_attempts > 0", name="ck_jobs_attempts"),
        # A running job always has a lease; no other status has one.
        sa.CheckConstraint(
            "(status = 'running') = (lease_token IS NOT NULL) "
            "AND (status = 'running') = (lease_expires_at IS NOT NULL) "
            "AND (status = 'running') = (worker_id IS NOT NULL)",
            name="ck_jobs_lease",
        ),
        sa.CheckConstraint(
            "(status IN ('completed', 'failed', 'cancelled')) = (finished_at IS NOT NULL)",
            name="ck_jobs_finished_at",
        ),
        schema=SCHEMA,
    )
    # The claim query: the oldest available queued job.
    op.create_index(
        "ix_jobs_claim",
        "jobs",
        ["available_at", "created_at"],
        schema=SCHEMA,
        postgresql_where=sa.text("status = 'queued'"),
    )
    # The lease sweep: running jobs by lease expiry.
    op.create_index(
        "ix_jobs_lease",
        "jobs",
        ["lease_expires_at"],
        schema=SCHEMA,
        postgresql_where=sa.text("status = 'running'"),
    )
    op.create_index(
        "uq_jobs_one_active_per_dedupe_key",
        "jobs",
        ["dedupe_key"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_table("jobs", schema=SCHEMA)
