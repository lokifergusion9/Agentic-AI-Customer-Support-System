"""Create order_items.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db import SCHEMA

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "order_items",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "order_id",
            sa.Text(),
            sa.ForeignKey(f"{SCHEMA}.orders.id", name="fk_order_items_order_id"),
            nullable=False,
        ),
        sa.Column("product_id", sa.Text(), nullable=False),
        sa.Column("product_name", sa.Text(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        # Line total in minor units (paise).
        sa.Column("paid_amount_minor", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("quantity > 0", name="ck_order_items_quantity"),
        sa.CheckConstraint("paid_amount_minor >= 0", name="ck_order_items_paid_amount_minor"),
        schema=SCHEMA,
    )
    op.create_index("ix_order_items_order_id", "order_items", ["order_id"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_table("order_items", schema=SCHEMA)
