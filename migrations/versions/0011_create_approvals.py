"""Create approvals.

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db import SCHEMA

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ROLES = "('reviewer', 'senior_reviewer')"
# Decisions made by an authenticated reviewer; expired and invalidated are set by the system.
REVIEWER_DECISION = "COALESCE(decision IN ('approved', 'rejected'), false)"


def upgrade() -> None:
    op.create_table(
        "approvals",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("proposal_id", sa.Text(), nullable=False),
        # The exact proposal this approval is for (W1.4); must equal the proposal's hash.
        sa.Column("approved_payload_hash", sa.Text(), nullable=False),
        # senior_reviewer for refunds above ₹50,000 (human_review_exceptions_v1:section_3).
        sa.Column("required_role", sa.Text(), nullable=False),
        # Copied from the reviewer's JWT when they decide (Q5); never from the model.
        sa.Column("reviewer_id", sa.Text(), nullable=True),
        sa.Column("reviewer_role", sa.Text(), nullable=True),
        sa.Column("decision", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["proposal_id", "approved_payload_hash"],
            [f"{SCHEMA}.action_proposals.id", f"{SCHEMA}.action_proposals.payload_hash"],
            name="fk_approvals_proposal_hash",
        ),
        # Target of the composite foreign key from refund_operations.
        sa.UniqueConstraint("id", "proposal_id", name="uq_approvals_id_proposal_id"),
        sa.CheckConstraint(f"required_role IN {ROLES}", name="ck_approvals_required_role"),
        sa.CheckConstraint(f"reviewer_role IN {ROLES}", name="ck_approvals_reviewer_role"),
        sa.CheckConstraint(
            "decision IN ('approved', 'rejected', 'expired', 'invalidated')",
            name="ck_approvals_decision",
        ),
        sa.CheckConstraint(
            "(decision IS NULL) = (decided_at IS NULL)", name="ck_approvals_decided_at"
        ),
        # Reviewer identity is recorded for reviewer decisions only.
        sa.CheckConstraint(
            f"{REVIEWER_DECISION} = (reviewer_id IS NOT NULL) "
            f"AND {REVIEWER_DECISION} = (reviewer_role IS NOT NULL)",
            name="ck_approvals_reviewer_fields",
        ),
        # An expired approval cannot be approved or rejected (W3.6).
        sa.CheckConstraint(
            f"NOT {REVIEWER_DECISION} OR decided_at <= expires_at",
            name="ck_approvals_decided_before_expiry",
        ),
        sa.CheckConstraint(
            "NOT (decision = 'approved' AND required_role = 'senior_reviewer' "
            "AND reviewer_role <> 'senior_reviewer')",
            name="ck_approvals_senior_reviewer",
        ),
        schema=SCHEMA,
    )
    # At most one open approval per proposal.
    op.create_index(
        "uq_approvals_one_pending_per_proposal",
        "approvals",
        ["proposal_id"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("decision IS NULL"),
    )


def downgrade() -> None:
    op.drop_table("approvals", schema=SCHEMA)
