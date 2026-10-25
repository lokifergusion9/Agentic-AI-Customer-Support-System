"""Create refund_operations.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.db import SCHEMA

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "refund_operations",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("order_id", sa.Text(), nullable=False),
        sa.Column("proposal_id", sa.Text(), nullable=False),
        sa.Column("approval_id", sa.Text(), nullable=False),
        sa.Column("amount_minor", sa.BigInteger(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("provider_reference", sa.Text(), nullable=True),
        # reserved, submitted and unknown hold part of the refundable balance; failed releases it.
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("provider_response", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        # The refund is for the proposal's order, under an approval of that same proposal.
        sa.ForeignKeyConstraint(
            ["proposal_id", "order_id"],
            [f"{SCHEMA}.action_proposals.id", f"{SCHEMA}.action_proposals.order_id"],
            name="fk_refund_operations_proposal_order",
        ),
        sa.ForeignKeyConstraint(
            ["approval_id", "proposal_id"],
            [f"{SCHEMA}.approvals.id", f"{SCHEMA}.approvals.proposal_id"],
            name="fk_refund_operations_approval_proposal",
        ),
        # One refund operation per proposal, with a stable idempotency key (W1.4, W3.9).
        sa.UniqueConstraint("proposal_id", name="uq_refund_operations_proposal_id"),
        sa.UniqueConstraint("idempotency_key", name="uq_refund_operations_idempotency_key"),
        sa.UniqueConstraint("provider_reference", name="uq_refund_operations_provider_reference"),
        sa.CheckConstraint("amount_minor > 0", name="ck_refund_operations_amount_minor"),
        sa.CheckConstraint(
            "status IN ('reserved', 'submitted', 'succeeded', 'failed', 'unknown')",
            name="ck_refund_operations_status",
        ),
        sa.CheckConstraint(
            "status <> 'succeeded' OR provider_reference IS NOT NULL",
            name="ck_refund_operations_succeeded_reference",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_refund_operations_order_id", "refund_operations", ["order_id"], schema=SCHEMA
    )
    # Safety net behind the app's own check (W1.14, W3.9): only an approved approval can back
    # a new refund operation.
    op.execute(
        f"""
        CREATE FUNCTION {SCHEMA}.check_refund_approval() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM {SCHEMA}.approvals
                WHERE id = NEW.approval_id AND decision = 'approved'
            ) THEN
                RAISE EXCEPTION 'approval % is not approved', NEW.approval_id
                    USING ERRCODE = 'check_violation';
            END IF;
            RETURN NEW;
        END
        $$
        """
    )
    op.execute(
        f"CREATE TRIGGER trg_refund_operations_approval BEFORE INSERT "
        f"ON {SCHEMA}.refund_operations "
        f"FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.check_refund_approval()"
    )
    # Safety net: committed refunds (succeeded plus reserved) never exceed the paid amount.
    # Locking the order row serialises concurrent refunds for the same order (W3.10).
    op.execute(
        f"""
        CREATE FUNCTION {SCHEMA}.check_refundable_balance() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE
            paid bigint;
            committed bigint;
        BEGIN
            IF NEW.status = 'failed' THEN
                RETURN NEW;
            END IF;
            SELECT paid_amount_minor INTO paid
            FROM {SCHEMA}.orders WHERE id = NEW.order_id FOR UPDATE;
            SELECT COALESCE(sum(amount_minor), 0) INTO committed
            FROM {SCHEMA}.refund_operations
            WHERE order_id = NEW.order_id AND status <> 'failed' AND id <> NEW.id;
            IF committed + NEW.amount_minor > paid THEN
                RAISE EXCEPTION 'refund of % exceeds refundable balance % for order %',
                    NEW.amount_minor, paid - committed, NEW.order_id
                    USING ERRCODE = 'check_violation';
            END IF;
            RETURN NEW;
        END
        $$
        """
    )
    op.execute(
        f"CREATE TRIGGER trg_refund_operations_balance BEFORE INSERT OR UPDATE "
        f"ON {SCHEMA}.refund_operations "
        f"FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.check_refundable_balance()"
    )


def downgrade() -> None:
    op.drop_table("refund_operations", schema=SCHEMA)
    op.execute(f"DROP FUNCTION {SCHEMA}.check_refundable_balance()")
    op.execute(f"DROP FUNCTION {SCHEMA}.check_refund_approval()")
