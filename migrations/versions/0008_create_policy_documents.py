"""Create policy_documents.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db import SCHEMA

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Needed for the exclusion constraint below (equality on text inside a GiST index).
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist WITH SCHEMA public")
    op.create_table(
        "policy_documents",
        # Document ID plus version, e.g. refund_policy_v1; the prefix of every citation.
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("policy_key", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("source_uri", sa.Text(), nullable=False),
        # Lets ingestion detect whether a source file changed (W2.1).
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("policy_key", "version", name="uq_policy_documents_key_version"),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_to >= effective_from",
            name="ck_policy_documents_effective_range",
        ),
        schema=SCHEMA,
    )
    # On any date, at most one version of each policy is in effect.
    op.execute(
        f"ALTER TABLE {SCHEMA}.policy_documents "
        "ADD CONSTRAINT ex_policy_documents_no_overlap EXCLUDE USING gist "
        "(policy_key WITH =, daterange(effective_from, effective_to, '[]') WITH &&)"
    )


def downgrade() -> None:
    op.drop_table("policy_documents", schema=SCHEMA)
    op.execute("DROP EXTENSION IF EXISTS btree_gist")
