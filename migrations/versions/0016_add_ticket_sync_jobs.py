"""Allow ticket_sync jobs: a ticket update the helpdesk has not confirmed yet (W3.14).

When the ticketing API fails, only the ticket update is retried, as its own job; the refund
or the rest of the run is never repeated for it.

Revision ID: 0016
Revises: 0015
Create Date: 2026-10-28
"""

from collections.abc import Sequence

from alembic import op

from app.db import SCHEMA

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("ck_jobs_job_type", "jobs", schema=SCHEMA, type_="check")
    op.create_check_constraint(
        "ck_jobs_job_type",
        "jobs",
        "job_type IN ('workflow_run', 'eval_attempt', 'ticket_sync')",
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.execute(f"DELETE FROM {SCHEMA}.jobs WHERE job_type = 'ticket_sync'")
    op.drop_constraint("ck_jobs_job_type", "jobs", schema=SCHEMA, type_="check")
    op.create_check_constraint(
        "ck_jobs_job_type", "jobs", "job_type IN ('workflow_run', 'eval_attempt')", schema=SCHEMA
    )
