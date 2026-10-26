"""Refund proposals and their pending approvals (W1.15 basic; W3.1, W3.2).

A proposal is immutable once stored (a database trigger forbids edits). The approval is bound
to the proposal's payload hash, so approving one proposal can never authorize another.
Proposals are only made from an `eligible` decision; the decision is never the model's.
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
from app.services.runs import get_run, set_run_status


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
    settings = get_settings()
    payload = {
        "action": "request_refund_approval",
        "order_id": decision.order_id,
        "amount_minor": decision.amount_minor,
        "currency": settings.store_currency,
        "reason": reason,
        "evidence_refs": evidence_refs,
        "policy_refs": decision.policy_refs,
        "missing_information": [],
    }
    digest = payload_hash(payload)

    existing = find_proposal(db, run_id, digest)
    if existing is not None:
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
    expires_at = (now or datetime.now(UTC)) + timedelta(minutes=settings.approval_expiry_minutes)
    db.execute(
        text(
            f"INSERT INTO {SCHEMA}.approvals (id, proposal_id, approved_payload_hash, "
            "required_role, expires_at) VALUES (:id, :proposal, :hash, :role, :expires)"
        ),
        {
            "id": f"apr_{secrets.token_hex(8)}",
            "proposal": proposal_id,
            "hash": digest,
            "role": decision.reviewer_role,
            "expires": expires_at,
        },
    )
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


def find_proposal(db: Session, run_id: str, digest: str) -> ProposalView | None:
    """The proposal with this payload in this run, with its latest approval."""
    row = db.execute(
        text(
            "SELECT p.id AS proposal_id, p.run_id, p.order_id, p.payload, p.payload_hash, "
            "p.status, a.id AS approval_id, a.required_role, a.expires_at "
            f"FROM {SCHEMA}.action_proposals p JOIN {SCHEMA}.approvals a ON a.proposal_id = p.id "
            "WHERE p.run_id = :run AND p.payload_hash = :hash "
            "ORDER BY a.created_at DESC LIMIT 1"
        ),
        {"run": run_id, "hash": digest},
    ).one_or_none()
    return None if row is None else ProposalView.model_validate(row._mapping)
