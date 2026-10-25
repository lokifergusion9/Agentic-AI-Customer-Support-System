"""Create orders.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db import SCHEMA

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "orders",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "customer_id",
            sa.Text(),
            sa.ForeignKey(f"{SCHEMA}.customers.id", name="fk_orders_customer_id"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False),
        # Only the format is checked here; the app enforces STORE_CURRENCY (Q6).
        sa.Column("currency", sa.CHAR(3), nullable=False),
        # Money is stored in integer minor units (paise). Paid amount includes shipping.
        sa.Column("paid_amount_minor", sa.BigInteger(), nullable=False),
        sa.Column("shipping_amount_minor", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("placed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        # Target of composite foreign keys that tie a row to an order *and* its owner.
        sa.UniqueConstraint("id", "customer_id", name="uq_orders_id_customer_id"),
        sa.CheckConstraint(
            "status IN ('placed', 'shipped', 'delivered', 'cancelled')", name="ck_orders_status"
        ),
        sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name="ck_orders_currency"),
        sa.CheckConstraint("paid_amount_minor >= 0", name="ck_orders_paid_amount_minor"),
        sa.CheckConstraint("shipping_amount_minor >= 0", name="ck_orders_shipping_amount_minor"),
        sa.CheckConstraint(
            "(status = 'delivered') = (delivered_at IS NOT NULL)", name="ck_orders_delivered_at"
        ),
        schema=SCHEMA,
    )
    op.create_index("ix_orders_customer_id", "orders", ["customer_id"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_table("orders", schema=SCHEMA)
