"""Reviewer decisions on pending approvals (W1.15 basic; W3.4-W3.7).

The reviewer comes from authentication, never from the request body, the chat or the model.
An approval can be decided once, before it expires, by a reviewer with the required role.
"""

from datetime import UTC, datetime

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.reviewer import REVIEWER_ROLES, AuthenticatedReviewer, can_decide
from app.db import SCHEMA
from app.services.audit import record_event
from app.services.errors import ServiceFailure
from app.services.runs import set_run_status

DECISIONS = ("approved", "rejected")


class ApprovalView(BaseModel):
    approval_id: str
    proposal_id: str
    required_role: str
    decision: str | None
    reviewer_id: str | None
    reviewer_role: str | None
    expires_at: datetime
    decided_at: datetime | None


APPROVAL_COLUMNS = (
    "id AS approval_id, proposal_id, required_role, decision, reviewer_id, reviewer_role, "
    "expires_at, decided_at"
)
PENDING_LIMIT = 50


class PendingApprovalView(BaseModel):
    """What a reviewer needs to decide one approval (W3.4, the W3.8 screen)."""

    approval_id: str
    proposal_id: str
    run_id: str
    ticket_id: str
    customer_id: str
    customer_name: str
    order_id: str
    complaint_category: str | None
    complaint_summary: str | None
    action: str
    amount_minor: int
    currency: str
    reason: str
    evidence_refs: list[str]
    policy_refs: list[str]
    required_role: str
    created_at: datetime
    expires_at: datetime


def list_pending_approvals(
    db: Session, reviewer: AuthenticatedReviewer, *, now: datetime | None = None
) -> list[PendingApprovalView]:
    """Undecided, unexpired approvals this reviewer may decide, the soonest to expire first.

    Senior-only approvals are left out for a standard reviewer: they stay queued for a senior.
    """
    roles = [role for role in REVIEWER_ROLES if can_decide(reviewer, role)]
    rows = db.execute(
        text(
            "SELECT a.id AS approval_id, a.proposal_id, p.run_id, r.ticket_id, "
            "c.id AS customer_id, c.name AS customer_name, p.order_id, "
            "t.category AS complaint_category, t.summary AS complaint_summary, p.payload, "
            "a.required_role, a.created_at, a.expires_at "
            f"FROM {SCHEMA}.approvals a "
            f"JOIN {SCHEMA}.action_proposals p "
            "ON p.id = a.proposal_id AND p.payload_hash = a.approved_payload_hash "
            f"JOIN {SCHEMA}.workflow_runs r ON r.id = p.run_id "
            f"JOIN {SCHEMA}.tickets t ON t.id = r.ticket_id "
            f"JOIN {SCHEMA}.customers c ON c.id = p.customer_id "
            "WHERE a.decision IS NULL AND a.expires_at >= :now "
            "AND a.required_role = ANY(:roles) "
            "ORDER BY a.expires_at, a.id LIMIT :limit"
        ),
        {"now": now or datetime.now(UTC), "roles": roles, "limit": PENDING_LIMIT},
    ).all()
    views = []
    for row in rows:
        fields = dict(row._mapping)
        payload = fields.pop("payload")
        views.append(
            PendingApprovalView(
                **fields,
                **{
                    key: payload[key]
                    for key in (
                        "action",
                        "amount_minor",
                        "currency",
                        "reason",
                        "evidence_refs",
                        "policy_refs",
                    )
                },
            )
        )
    return views


def decide_approval(
    db: Session,
    reviewer: AuthenticatedReviewer,
    approval_id: str,
    decision: str,
    *,
    now: datetime | None = None,
) -> ApprovalView:
    if decision not in DECISIONS:
        raise ValueError(f"unknown decision {decision!r}")
    now = now or datetime.now(UTC)

    # Lock the approval so two reviewers deciding at once cannot both succeed.
    approval = db.execute(
        text(
            "SELECT a.id, a.proposal_id, a.required_role, a.decision, a.expires_at, "
            "p.run_id, r.ticket_id "
            f"FROM {SCHEMA}.approvals a "
            f"JOIN {SCHEMA}.action_proposals p "
            "ON p.id = a.proposal_id AND p.payload_hash = a.approved_payload_hash "
            f"JOIN {SCHEMA}.workflow_runs r ON r.id = p.run_id "
            "WHERE a.id = :approval FOR UPDATE OF a"
        ),
        {"approval": approval_id},
    ).one_or_none()
    if approval is None:
        raise ServiceFailure("APPROVAL_NOT_FOUND", "No approval with this ID was found.")
    if approval.decision is not None:
        raise ServiceFailure(
            "APPROVAL_ALREADY_DECIDED", f"This approval is already {approval.decision}."
        )

    if now > approval.expires_at:
        # The system records the expiry; a new approval must be requested (W3.6).
        close(db, approval, "expired", now, actor="system:approvals")
        db.commit()
        raise ServiceFailure("APPROVAL_EXPIRED", "This approval expired; request a new one.")

    if not can_decide(reviewer, approval.required_role):
        raise ServiceFailure(
            "REVIEWER_NOT_PERMITTED",
            "A senior reviewer must decide this approval.",
        )

    close(db, approval, decision, now, actor=f"reviewer:{reviewer.reviewer_id}", reviewer=reviewer)
    # The run goes on either way: the resumed workflow executes the refund or follows the
    # rejection path, and closes the run itself (W3.3).
    set_run_status(db, approval.run_id, "running", actor=f"reviewer:{reviewer.reviewer_id}")
    db.commit()
    return get_approval(db, approval_id)


def close(
    db: Session,
    approval,
    decision: str,
    now: datetime,
    *,
    actor: str,
    reviewer: AuthenticatedReviewer | None = None,
) -> None:
    """Record the decision on the approval and its proposal, in the caller's transaction."""
    db.execute(
        text(
            f"UPDATE {SCHEMA}.approvals SET decision = :decision, decided_at = :now, "
            "reviewer_id = :reviewer_id, reviewer_role = :reviewer_role WHERE id = :id"
        ),
        {
            "id": approval.id,
            "decision": decision,
            "now": now,
            "reviewer_id": reviewer.reviewer_id if reviewer else None,
            "reviewer_role": reviewer.role if reviewer else None,
        },
    )
    db.execute(
        text(f"UPDATE {SCHEMA}.action_proposals SET status = :status WHERE id = :id"),
        {"id": approval.proposal_id, "status": decision},
    )
    record_event(
        db,
        actor=actor,
        event_type="approval_decided",
        payload={
            "approval_id": approval.id,
            "proposal_id": approval.proposal_id,
            "decision": decision,
            "reviewer_role": reviewer.role if reviewer else None,
        },
        ticket_id=approval.ticket_id,
        run_id=approval.run_id,
    )


def get_approval(db: Session, approval_id: str) -> ApprovalView:
    row = db.execute(
        text(f"SELECT {APPROVAL_COLUMNS} FROM {SCHEMA}.approvals WHERE id = :id"),
        {"id": approval_id},
    ).one_or_none()
    if row is None:
        raise ServiceFailure("APPROVAL_NOT_FOUND", "No approval with this ID was found.")
    return ApprovalView.model_validate(row._mapping)
