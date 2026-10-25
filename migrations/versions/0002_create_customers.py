"""Create customers.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db import SCHEMA

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "customers",
        sa.Column("id", sa.Text(), primary_key=True),
        # The JWT `sub` claim maps to this column (W1.5).
        sa.Column("external_auth_subject", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("external_auth_subject", name="uq_customers_external_auth_subject"),
        sa.UniqueConstraint("email", name="uq_customers_email"),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("customers", schema=SCHEMA)
