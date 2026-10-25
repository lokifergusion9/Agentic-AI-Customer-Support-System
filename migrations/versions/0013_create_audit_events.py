"""Create audit_events.

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.db import SCHEMA

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_events",
        # Increasing ID gives the run timeline its order (GET /runs/{run_id}/events).
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column(
            "run_id",
            sa.Text(),
            sa.ForeignKey(f"{SCHEMA}.workflow_runs.id", name="fk_audit_events_run_id"),
            nullable=True,
        ),
        # For events before a run exists, such as ticket creation.
        sa.Column(
            "ticket_id",
            sa.Text(),
            sa.ForeignKey(f"{SCHEMA}.tickets.id", name="fk_audit_events_ticket_id"),
            nullable=True,
        ),
        # e.g. customer:cus_alice, reviewer:rev_1, agent:investigation, system:worker
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column(
            "redacted_payload",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "actor ~ '^(customer|reviewer|agent|system):.+'", name="ck_audit_events_actor"
        ),
        schema=SCHEMA,
    )
    op.create_index("ix_audit_events_run_id", "audit_events", ["run_id", "id"], schema=SCHEMA)
    op.create_index("ix_audit_events_ticket_id", "audit_events", ["ticket_id"], schema=SCHEMA)
    # The audit trail is append-only.
    op.execute(
        f"""
        CREATE FUNCTION {SCHEMA}.forbid_audit_event_change() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'audit_events is append-only'
                USING ERRCODE = 'integrity_constraint_violation';
        END
        $$
        """
    )
    op.execute(
        f"CREATE TRIGGER trg_audit_events_append_only BEFORE UPDATE OR DELETE "
        f"ON {SCHEMA}.audit_events "
        f"FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.forbid_audit_event_change()"
    )


def downgrade() -> None:
    op.drop_table("audit_events", schema=SCHEMA)
    op.execute(f"DROP FUNCTION {SCHEMA}.forbid_audit_event_change()")
