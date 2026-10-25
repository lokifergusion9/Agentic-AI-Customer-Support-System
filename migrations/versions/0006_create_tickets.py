"""Create tickets.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db import SCHEMA

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tickets",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "customer_id",
            sa.Text(),
            sa.ForeignKey(f"{SCHEMA}.customers.id", name="fk_tickets_customer_id"),
            nullable=False,
        ),
        # Empty until the customer names the order (missing-information loop, W2.13).
        sa.Column("order_id", sa.Text(), nullable=True),
        # Empty until the complaint is classified (W2.8).
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("customer_message", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        # Ticket ID in the external ticketing API (Q1).
        sa.Column("external_ticket_ref", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        # A ticket can only reference an order owned by the same customer.
        sa.ForeignKeyConstraint(
            ["order_id", "customer_id"],
            [f"{SCHEMA}.orders.id", f"{SCHEMA}.orders.customer_id"],
            name="fk_tickets_order_owner",
        ),
        sa.UniqueConstraint("external_ticket_ref", name="uq_tickets_external_ticket_ref"),
        sa.CheckConstraint(
            "category IN ('order_status', 'refund_request', 'damaged_item', 'out_of_scope')",
            name="ck_tickets_category",
        ),
        sa.CheckConstraint(
            "status IN ('open', 'awaiting_customer', 'awaiting_approval', 'escalated', 'resolved')",
            name="ck_tickets_status",
        ),
        schema=SCHEMA,
    )
    op.create_index("ix_tickets_customer_id", "tickets", ["customer_id"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_table("tickets", schema=SCHEMA)
