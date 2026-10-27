"""Add workflow_runs.output.

The run's result as the workflow last left it: final response, route, category, proposal and
approval IDs, citations and stop reason. GET /runs/{run_id} serves it (W2.14). Redacted the same
way as the audit trail; the customer's own message stays in tickets.

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.db import SCHEMA

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "workflow_runs",
        sa.Column("output", postgresql.JSONB(), nullable=True),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("workflow_runs", "output", schema=SCHEMA)
