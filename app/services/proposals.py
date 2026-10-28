"""Refund proposals and their pending approvals (W1.15 basic; W3.1, W3.2, W3.6).

A proposal is immutable once stored (a database trigger forbids edits). The approval is bound
to the proposal's payload hash, so approving one proposal can never authorize another.
Proposals are only made from an `eligible` decision; the decision is never the model's.

When facts change before the refund runs, the proposal is invalidated and a new one (new
payload, new hash) needs its own approval. An expired approval of an unchanged proposal is
renewed: a new pending approval for the same immutable proposal (W3.6).
"""

import hashlib
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.config import get_settings
from app.db import SCHEMA
from app.services.audit import record_event
from app.services.eligibility import EligibilityDecision
from app.services.errors import ServiceFailure
from app.services.runs import get_run, get_run_for_reviewer, set_run_status


def payload_hash(payload: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON form of a proposal payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ProposalView(BaseModel):
    proposal_id: str
    run_id: str
    order_id: str
    payload: dict[str, Any]
    payload_hash: str
    status: str
    approval_id: str
    required_role: str
    expires_at: datetime


def refund_payload(
    decision: EligibilityDecision, *, reason: str, evidence_refs: list[str]
) -> dict[str, Any]:
    """The proposal payload for an eligible decision; its hash is what a reviewer approves."""
    return {
        "action": "request_refund_approval",
        "order_id": decision.order_id,
        "amount_minor": decision.amount_minor,
        "currency": get_settings().store_currency,
        "reason": reason,
        "evidence_refs": evidence_refs,
        "policy_refs": decision.policy_refs,
        "missing_information": [],
    }


def create_refund_proposal(
    db: Session,
    customer: AuthenticatedCustomer,
    run_id: str,
    decision: EligibilityDecision,
    *,
    reason: str,
    evidence_refs: list[str],
    now: datetime | None = None,
) -> ProposalView:
    """Store a refund proposal and a pending approval for it; the run waits for approval.

    Asking again with the same payload in the same run returns the stored proposal, so a
    restarted step cannot create a second one.
    """
    if decision.outcome != "eligible" or decision.amount_minor is None:
        raise ServiceFailure(
            "NOT_ELIGIBLE", f"The order is not eligible for a refund ({decision.reason})."
        )
    run = get_run(db, customer, run_id)  # RUN_NOT_FOUND unless it is the customer's
    payload = refund_payload(decision, reason=reason, evidence_refs=evidence_refs)
    digest = payload_hash(payload)

    existing = find_proposal(db, run_id, digest)
    if existing is not None:
        if existing.status == "invalidated":
            # The facts went back to an earlier state; a person should look at this run.
            raise ServiceFailure(
                "PROPOSAL_INVALIDATED", "This refund was already proposed and invalidated."
            )
        return existing

    proposal_id = f"prp_{secrets.token_hex(8)}"
    db.execute(
        text(
            f"INSERT INTO {SCHEMA}.action_proposals (id, run_id, order_id, customer_id, "
            "action_type, payload, payload_hash, status) VALUES (:id, :run, :order, :customer, "
            "'refund', CAST(:payload AS jsonb), :hash, 'pending_approval')"
        ),
        {
            "id": proposal_id,
            "run": run_id,
            "order": decision.order_id,
            "customer": customer.customer_id,
            "payload": json.dumps(payload),
            "hash": digest,
        },
    )
    assert decision.reviewer_role is not None
    insert_pending_approval(db, proposal_id, digest, decision.reviewer_role, now)
    actor = f"customer:{customer.customer_id}"
    set_run_status(db, run_id, "waiting_for_approval", actor=actor)
    record_event(
        db,
        actor=actor,
        event_type="proposal_created",
        payload={
            "proposal_id": proposal_id,
            "order_id": decision.order_id,
            "amount_minor": decision.amount_minor,
            "required_role": decision.reviewer_role,
        },
        ticket_id=run.ticket_id,
        run_id=run_id,
    )
    db.commit()
    proposal = find_proposal(db, run_id, digest)
    assert proposal is not None
    return proposal


def renew_approval(
    db: Session, proposal_id: str, *, actor: str, now: datetime | None = None
) -> ProposalView:
    """A new pending approval for an unchanged proposal whose approval expired (W3.6).

    Renewing again returns the open approval, so a restarted step cannot add a second one.
    """
    proposal = get_proposal(db, proposal_id)
    if proposal.status == "pending_approval":
        return proposal
    if proposal.status != "expired":
        raise ServiceFailure(
            "APPROVAL_NOT_EXPIRED", f"Only an expired approval is renewed, not {proposal.status}."
        )
    insert_pending_approval(db, proposal_id, proposal.payload_hash, proposal.required_role, now)
    set_status(db, proposal_id, "pending_approval")
    run = set_run_status(db, proposal.run_id, "waiting_for_approval", actor=actor)
    record_event(
        db,
        actor=actor,
        event_type="approval_renewed",
        payload={"proposal_id": proposal_id, "expired_approval_id": proposal.approval_id},
        ticket_id=run.ticket_id,
        run_id=proposal.run_id,
    )
    db.commit()
    return get_proposal(db, proposal_id)


def invalidate_proposal(db: Session, proposal_id: str, *, reason: str, actor: str) -> None:
    """The facts changed: this proposal, and any approval of it, can never be executed (W3.6).

    The reviewer's decision stays on the approval as it was made. Execution needs the
    proposal to be `approved`, so an invalidated proposal is refused even if it was approved.
    """
    proposal = get_proposal(db, proposal_id)
    if proposal.status == "executed":
        raise ServiceFailure("PROPOSAL_EXECUTED", "This refund was already executed.")
    db.execute(
        text(
            f"UPDATE {SCHEMA}.approvals SET decision = 'invalidated', decided_at = :now "
            "WHERE proposal_id = :proposal AND decision IS NULL"
        ),
        {"proposal": proposal_id, "now": datetime.now(UTC)},
    )
    set_status(db, proposal_id, "invalidated")
    record_event(
        db,
        actor=actor,
        event_type="proposal_invalidated",
        payload={"proposal_id": proposal_id, "previous_status": proposal.status, "reason": reason},
        ticket_id=get_run_for_reviewer(db, proposal.run_id).ticket_id,
        run_id=proposal.run_id,
    )
    db.commit()


def insert_pending_approval(
    db: Session, proposal_id: str, digest: str, role: str, now: datetime | None
) -> str:
    approval_id = f"apr_{secrets.token_hex(8)}"
    expiry = timedelta(minutes=get_settings().approval_expiry_minutes)
    db.execute(
        text(
            f"INSERT INTO {SCHEMA}.approvals (id, proposal_id, approved_payload_hash, "
            "required_role, expires_at) VALUES (:id, :proposal, :hash, :role, :expires)"
        ),
        {
            "id": approval_id,
            "proposal": proposal_id,
            "hash": digest,
            "role": role,
            "expires": (now or datetime.now(UTC)) + expiry,
        },
    )
    return approval_id


def set_status(db: Session, proposal_id: str, status: str) -> None:
    db.execute(
        text(f"UPDATE {SCHEMA}.action_proposals SET status = :status WHERE id = :id"),
        {"id": proposal_id, "status": status},
    )


PROPOSAL_QUERY = (
    "SELECT p.id AS proposal_id, p.run_id, p.order_id, p.payload, p.payload_hash, "
    "p.status, a.id AS approval_id, a.required_role, a.expires_at "
    f"FROM {SCHEMA}.action_proposals p JOIN {SCHEMA}.approvals a ON a.proposal_id = p.id "
)
# The latest approval: an open one first, then the newest.
LATEST_APPROVAL = "ORDER BY a.decision IS NULL DESC, a.created_at DESC LIMIT 1"


def find_proposal(db: Session, run_id: str, digest: str) -> ProposalView | None:
    """The proposal with this payload in this run, with its latest approval."""
    row = db.execute(
        text(f"{PROPOSAL_QUERY}WHERE p.run_id = :run AND p.payload_hash = :hash {LATEST_APPROVAL}"),
        {"run": run_id, "hash": digest},
    ).one_or_none()
    return None if row is None else ProposalView.model_validate(row._mapping)


def get_proposal(db: Session, proposal_id: str) -> ProposalView:
    """One proposal with its latest approval."""
    row = db.execute(
        text(f"{PROPOSAL_QUERY}WHERE p.id = :id {LATEST_APPROVAL}"), {"id": proposal_id}
    ).one_or_none()
    if row is None:
        raise ServiceFailure("PROPOSAL_NOT_FOUND", "No refund proposal with this ID was found.")
    return ProposalView.model_validate(row._mapping)
