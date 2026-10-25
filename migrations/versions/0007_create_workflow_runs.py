"""Create workflow_runs.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db import SCHEMA

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TERMINAL_STATUSES = "('completed', 'escalated', 'failed')"


def upgrade() -> None:
    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "ticket_id",
            sa.Text(),
            sa.ForeignKey(f"{SCHEMA}.tickets.id", name="fk_workflow_runs_ticket_id"),
            nullable=False,
        ),
        # LangGraph thread ID; resume uses the same thread (W3.3).
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("thread_id", name="uq_workflow_runs_thread_id"),
        sa.CheckConstraint(
            "status IN ('running', 'waiting_for_customer', 'waiting_for_approval', "
            "'completed', 'escalated', 'failed')",
            name="ck_workflow_runs_status",
        ),
        sa.CheckConstraint(
            f"(status IN {TERMINAL_STATUSES}) = (completed_at IS NOT NULL)",
            name="ck_workflow_runs_completed_at",
        ),
        schema=SCHEMA,
    )
    op.create_index("ix_workflow_runs_ticket_id", "workflow_runs", ["ticket_id"], schema=SCHEMA)
    # At most one unfinished run per ticket, so two runs cannot act on the same ticket.
    op.create_index(
        "uq_workflow_runs_one_active_per_ticket",
        "workflow_runs",
        ["ticket_id"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text(f"status NOT IN {TERMINAL_STATUSES}"),
    )


def downgrade() -> None:
    op.drop_table("workflow_runs", schema=SCHEMA)
