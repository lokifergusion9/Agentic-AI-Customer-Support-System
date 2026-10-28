"""Execute an approved refund, once (W1.15 basic; the core of W3.9).

1. Verify the stored approval for this exact proposal (never a chat message, W3.7).
2. Check the stored payload still hashes to the approved one, then recheck eligibility and
   the refundable balance (audited as `eligibility_rechecked`).
3. Reserve the amount in a refund operation with a stable idempotency key, and commit
   before calling the provider: no transaction is held open during the network call.
4. Call the provider with that key and persist what it said.

If the outcome is uncertain (timeout or a retryable error) the operation is left `unknown`.
Running this again resends the same key and payload, which the provider answers with the
same refund, so a retry never creates a second one. Full reconciliation is W3.12.
"""

import json
import secrets
from typing import Any

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.services.audit import record_event
from app.services.eligibility import check_refund_eligibility
from app.services.errors import ServiceFailure
from app.services.proposals import payload_hash
from app.services.providers import ProviderClient

ACTOR = "system:executor"
# Settled locally: nothing more to send for this operation.
SETTLED = {"succeeded", "failed", "submitted"}


class RefundOperationView(BaseModel):
    operation_id: str
    proposal_id: str
    order_id: str
    amount_minor: int
    idempotency_key: str
    # reserved | submitted | succeeded | failed | unknown
    status: str
    provider_reference: str | None
    provider_response: dict[str, Any] | None


OPERATION_COLUMNS = (
    "id AS operation_id, proposal_id, order_id, amount_minor, idempotency_key, status, "
    "provider_reference, provider_response"
)


def idempotency_key(proposal_id: str) -> str:
    """Stable for the proposal, so every retry sends the same key."""
    return f"refund:{proposal_id}"


def execute_approved_refund(
    db: Session, payments: ProviderClient, proposal_id: str
) -> RefundOperationView:
    proposal = db.execute(
        text(
            "SELECT p.id, p.run_id, p.order_id, p.customer_id, p.payload, p.payload_hash, "
            "p.status, r.ticket_id, c.external_auth_subject "
            f"FROM {SCHEMA}.action_proposals p "
            f"JOIN {SCHEMA}.workflow_runs r ON r.id = p.run_id "
            f"JOIN {SCHEMA}.customers c ON c.id = p.customer_id "
            "WHERE p.id = :proposal AND p.action_type = 'refund'"
        ),
        {"proposal": proposal_id},
    ).one_or_none()
    if proposal is None:
        raise ServiceFailure("PROPOSAL_NOT_FOUND", "No refund proposal with this ID was found.")

    existing = find_operation(db, proposal_id)
    if existing is not None:
        if existing.status in SETTLED:
            return existing
        # reserved or unknown: the provider may or may not have it; resend the same request.
        return send(db, payments, proposal, existing)

    approval_id = db.execute(
        text(
            f"SELECT id FROM {SCHEMA}.approvals WHERE proposal_id = :proposal "
            "AND decision = 'approved' AND approved_payload_hash = :hash"
        ),
        {"proposal": proposal_id, "hash": proposal.payload_hash},
    ).scalar_one_or_none()
    if approval_id is None or proposal.status != "approved":
        raise ServiceFailure(
            "APPROVAL_REQUIRED", "This proposal has no valid approval from a reviewer."
        )

    # Only the approved action runs: the stored payload must still hash to what was approved.
    if payload_hash(proposal.payload) != proposal.payload_hash:
        raise ServiceFailure(
            "APPROVAL_REQUIRED", "The proposal no longer matches the approved payload."
        )

    amount = proposal.payload["amount_minor"]
    customer = AuthenticatedCustomer(proposal.customer_id, proposal.external_auth_subject)
    decision = check_refund_eligibility(
        db, payments, customer, proposal.order_id, amount_minor=amount
    )
    record_event(
        db,
        actor=ACTOR,
        event_type="eligibility_rechecked",
        payload={
            "proposal_id": proposal_id,
            "outcome": decision.outcome,
            "reason": decision.reason,
            "amount_minor": amount,
            "refundable_balance_minor": decision.refundable_balance_minor,
        },
        ticket_id=proposal.ticket_id,
        run_id=proposal.run_id,
    )
    if decision.outcome != "eligible":
        db.commit()  # keep the recheck in the audit trail; nothing was reserved
        raise ServiceFailure(
            "NOT_ELIGIBLE", f"The order is no longer eligible for this refund ({decision.reason})."
        )

    # Reserve in a short transaction; the database trigger locks the order row and rejects
    # an amount above the refundable balance (W3.10).
    operation_id = f"rop_{secrets.token_hex(8)}"
    db.execute(
        text(
            f"INSERT INTO {SCHEMA}.refund_operations (id, order_id, proposal_id, approval_id, "
            "amount_minor, idempotency_key, status) "
            "VALUES (:id, :order, :proposal, :approval, :amount, :key, 'reserved')"
        ),
        {
            "id": operation_id,
            "order": proposal.order_id,
            "proposal": proposal_id,
            "approval": approval_id,
            "amount": amount,
            "key": idempotency_key(proposal_id),
        },
    )
    record_event(
        db,
        actor=ACTOR,
        event_type="refund_reserved",
        payload={"operation_id": operation_id, "proposal_id": proposal_id, "amount_minor": amount},
        ticket_id=proposal.ticket_id,
        run_id=proposal.run_id,
    )
    db.commit()

    operation = find_operation(db, proposal_id)
    assert operation is not None
    return send(db, payments, proposal, operation)


def send(
    db: Session, payments: ProviderClient, proposal, operation: RefundOperationView
) -> RefundOperationView:
    """Call the provider with the operation's stable key and persist the outcome."""
    reference: str | None = None
    try:
        body = payments.request(
            "POST",
            "/refunds",
            json={
                "idempotency_key": operation.idempotency_key,
                "order_id": operation.order_id,
                "amount_minor": operation.amount_minor,
                "currency": proposal.payload["currency"],
            },
        )
    except ServiceFailure as failure:
        # Retryable means the provider may have the refund: never assume it failed.
        status = "unknown" if failure.retryable else "failed"
        response: dict[str, Any] = {
            "error_code": failure.error_code,
            "retryable": failure.retryable,
            "message": failure.message,
        }
    else:
        provider_status = body.get("status")
        status = {"succeeded": "succeeded", "pending": "submitted"}.get(provider_status, "failed")
        reference = body.get("provider_reference")
        response = body

    db.execute(
        text(
            f"UPDATE {SCHEMA}.refund_operations SET status = :status, "
            "provider_reference = COALESCE(:reference, provider_reference), "
            "provider_response = CAST(:response AS jsonb), updated_at = now() WHERE id = :id"
        ),
        {
            "id": operation.operation_id,
            "status": status,
            "reference": reference,
            "response": json.dumps(response),
        },
    )
    if status == "succeeded":
        db.execute(
            text(f"UPDATE {SCHEMA}.action_proposals SET status = 'executed' WHERE id = :id"),
            {"id": operation.proposal_id},
        )
    record_event(
        db,
        actor=ACTOR,
        event_type="refund_executed",
        payload={
            "operation_id": operation.operation_id,
            "proposal_id": operation.proposal_id,
            "status": status,
            "amount_minor": operation.amount_minor,
            "error_code": response.get("error_code"),
        },
        ticket_id=proposal.ticket_id,
        run_id=proposal.run_id,
    )
    db.commit()
    result = find_operation(db, operation.proposal_id)
    assert result is not None
    return result


def find_operation(db: Session, proposal_id: str) -> RefundOperationView | None:
    row = db.execute(
        text(f"SELECT {OPERATION_COLUMNS} FROM {SCHEMA}.refund_operations WHERE proposal_id = :p"),
        {"p": proposal_id},
    ).one_or_none()
    return None if row is None else RefundOperationView.model_validate(row._mapping)
