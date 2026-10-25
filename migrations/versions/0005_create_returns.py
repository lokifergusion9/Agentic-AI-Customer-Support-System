"""Create returns.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db import SCHEMA

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "returns",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "order_id",
            sa.Text(),
            sa.ForeignKey(f"{SCHEMA}.orders.id", name="fk_returns_order_id"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False),
        # The refund window is measured from delivery to this date (refund_policy section 3).
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("inspection_status", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        # One return covers one order (return_processing section 2).
        sa.UniqueConstraint("order_id", name="uq_returns_order_id"),
        sa.CheckConstraint(
            "status IN ('requested', 'in_transit', 'received', 'closed')",
            name="ck_returns_status",
        ),
        sa.CheckConstraint(
            "inspection_status IN ('pending', 'passed', 'failed')",
            name="ck_returns_inspection_status",
        ),
        # A return counts as received only with a recorded received date (return_processing
        # section 3), and only a received item can have an inspection status (section 4).
        sa.CheckConstraint(
            "(status IN ('received', 'closed')) = (received_at IS NOT NULL)",
            name="ck_returns_received_at",
        ),
        sa.CheckConstraint(
            "(inspection_status IS NULL) = (received_at IS NULL)",
            name="ck_returns_inspection_after_receipt",
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("returns", schema=SCHEMA)
