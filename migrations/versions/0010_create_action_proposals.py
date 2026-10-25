"""Create action_proposals.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.db import SCHEMA

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "action_proposals",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "run_id",
            sa.Text(),
            sa.ForeignKey(f"{SCHEMA}.workflow_runs.id", name="fk_action_proposals_run_id"),
            nullable=False,
        ),
        sa.Column("order_id", sa.Text(), nullable=False),
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("action_type", sa.Text(), nullable=False),
        # The structured resolution proposal (W2.7).
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        # SHA-256 of the canonical payload JSON; approvals bind to it.
        sa.Column("payload_hash", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["order_id", "customer_id"],
            [f"{SCHEMA}.orders.id", f"{SCHEMA}.orders.customer_id"],
            name="fk_action_proposals_order_owner",
        ),
        # A restarted node (W3.3) cannot store the same proposal twice for one run.
        sa.UniqueConstraint("run_id", "payload_hash", name="uq_action_proposals_run_payload"),
        # Targets of composite foreign keys from approvals and refund_operations.
        sa.UniqueConstraint("id", "payload_hash", name="uq_action_proposals_id_payload_hash"),
        sa.UniqueConstraint("id", "order_id", name="uq_action_proposals_id_order_id"),
        sa.CheckConstraint(
            "action_type IN ('refund', 'replacement')", name="ck_action_proposals_action_type"
        ),
        sa.CheckConstraint(
            "payload_hash ~ '^[0-9a-f]{64}$'", name="ck_action_proposals_payload_hash"
        ),
        sa.CheckConstraint(
            "status IN ('pending_approval', 'approved', 'rejected', 'expired', 'invalidated', "
            "'executed')",
            name="ck_action_proposals_status",
        ),
        schema=SCHEMA,
    )
    op.create_index("ix_action_proposals_run_id", "action_proposals", ["run_id"], schema=SCHEMA)
    # Proposals are immutable (W3.1): only the status may change after creation.
    op.execute(
        f"""
        CREATE FUNCTION {SCHEMA}.forbid_action_proposal_change() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF (NEW.id, NEW.run_id, NEW.order_id, NEW.customer_id, NEW.action_type,
                NEW.payload, NEW.payload_hash, NEW.created_at)
               IS DISTINCT FROM
               (OLD.id, OLD.run_id, OLD.order_id, OLD.customer_id, OLD.action_type,
                OLD.payload, OLD.payload_hash, OLD.created_at) THEN
                RAISE EXCEPTION 'action proposal % is immutable; create a new proposal', OLD.id
                    USING ERRCODE = 'integrity_constraint_violation';
            END IF;
            RETURN NEW;
        END
        $$
        """
    )
    op.execute(
        f"CREATE TRIGGER trg_action_proposals_immutable BEFORE UPDATE ON {SCHEMA}.action_proposals "
        f"FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.forbid_action_proposal_change()"
    )


def downgrade() -> None:
    op.drop_table("action_proposals", schema=SCHEMA)
    op.execute(f"DROP FUNCTION {SCHEMA}.forbid_action_proposal_change()")
