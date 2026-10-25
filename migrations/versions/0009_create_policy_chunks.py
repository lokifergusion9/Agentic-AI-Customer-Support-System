"""Create policy_chunks.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.db import SCHEMA

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must match EMBEDDING_DIMENSIONS (all-MiniLM-L6-v2, Q3).
EMBEDDING_DIMENSIONS = 384


def upgrade() -> None:
    op.create_table(
        "policy_chunks",
        # One chunk per policy section; the ID is the citation, e.g. refund_policy_v1:section_3.
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "document_id",
            sa.Text(),
            sa.ForeignKey(
                f"{SCHEMA}.policy_documents.id",
                name="fk_policy_chunks_document_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("section_number", sa.Integer(), nullable=False),
        sa.Column("section_title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "document_id", "section_number", name="uq_policy_chunks_document_section"
        ),
        sa.CheckConstraint("section_number > 0", name="ck_policy_chunks_section_number"),
        schema=SCHEMA,
    )
    # SQLAlchemy has no built-in vector type; the pgvector Python package arrives in W2.1.
    op.execute(
        f"ALTER TABLE {SCHEMA}.policy_chunks "
        f"ADD COLUMN embedding public.vector({EMBEDDING_DIMENSIONS}) NOT NULL"
    )
    op.create_index("ix_policy_chunks_document_id", "policy_chunks", ["document_id"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_table("policy_chunks", schema=SCHEMA)
