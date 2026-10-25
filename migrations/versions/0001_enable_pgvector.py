"""Enable the pgvector extension (Q3).

Revision ID: 0001
Revises:
Create Date: 2026-10-25
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Extensions are database-wide; keep the vector type in public so every schema can use it.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
