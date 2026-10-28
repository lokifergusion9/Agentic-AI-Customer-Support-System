"""Execute an approved refund, once, and confirm it with the provider (W1.15; W3.9-W3.11).

1. Verify the stored approval for this exact proposal (never a chat message, W3.7).
2. Check the stored payload still hashes to the approved one, then recheck eligibility and
   the refundable balance (audited as `eligibility_rechecked`).
3. Reserve the amount in a refund operation with a stable idempotency key, and commit
   before calling the provider: no transaction is held open during the network call. The
   database trigger serializes reservations per order, so concurrent calls can never
   reserve more than the refundable balance (W3.10).
4. Call the provider with that key and persist what it said.

If the outcome is uncertain (timeout or a retryable error) the operation is left `unknown`.
Nothing is ever decided from local state alone: `confirm_refund` asks the provider by key
first, records what it already has, and resends the same key and payload only when it has
no record (W3.9 step 7). The provider answers a resend with the same refund, so a retry never
creates a second one.
"""

import json
import secrets
from typing import Any

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.services.audit import record_event
from app.services.eligibility import check_refund_eligibility
from app.services.errors import ServiceFailure
from app.services.proposals import payload_hash
from app.services.providers import ProviderClient

ACTOR = "system:executor"
# Final: the provider confirmed the outcome, nothing more to ask or send.
FINAL = {"succeeded", "failed"}
# The provider's refund status, as a local operation status.
FROM_PROVIDER = {"succeeded": "succeeded", "pending": "submitted", "failed": "failed"}
CHECK_VIOLATION = "23514"


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
    proposal = load_proposal(db, proposal_id)

    if find_operation(db, proposal_id) is not None:
        # Already reserved: ask the provider before sending anything again (W3.9 step 7).
        return confirm_refund(db, payments, proposal_id)

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
        if find_operation(db, proposal_id) is not None:
            # A concurrent call for this proposal reserved (and maybe paid) it meanwhile.
            return confirm_refund(db, payments, proposal_id)
        raise ServiceFailure(
            "NOT_ELIGIBLE", f"The order is no longer eligible for this refund ({decision.reason})."
        )

    # Reserve in a short transaction; the database trigger locks the order row and rejects
    # an amount above the refundable balance (W3.10).
    operation_id = f"rop_{secrets.token_hex(8)}"
    try:
        db.execute(
            text(
                f"INSERT INTO {SCHEMA}.refund_operations (id, order_id, proposal_id, "
                "approval_id, amount_minor, idempotency_key, status) "
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
            payload={
                "operation_id": operation_id,
                "proposal_id": proposal_id,
                "amount_minor": amount,
            },
            ticket_id=proposal.ticket_id,
            run_id=proposal.run_id,
        )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if find_operation(db, proposal_id) is not None:
            # A concurrent call reserved this proposal first: go on with its operation.
            return confirm_refund(db, payments, proposal_id)
        if getattr(exc.orig, "sqlstate", None) == CHECK_VIOLATION:
            # Another refund for the order took the balance between the recheck and now.
            raise ServiceFailure(
                "REFUND_EXCEEDS_BALANCE",
                "Another refund for this order used the refundable balance first.",
            ) from None
        raise

    operation = find_operation(db, proposal_id)
    assert operation is not None
    return send(db, payments, proposal, operation)


def confirm_refund(db: Session, payments: ProviderClient, proposal_id: str) -> RefundOperationView:
    """Bring an unfinished operation in line with the provider (W3.9 step 7, W3.11, W3.12).

    Final operations are returned as they are. Otherwise the provider is asked by idempotency
    key: what it has is recorded (`refund_reconciled`), and only when it has no record of a
    `reserved` or `unknown` operation is the same request sent again. If the provider cannot
    be reached, the operation stays as it is, to be confirmed later.
    """
    operation = find_operation(db, proposal_id)
    if operation is None:
        raise ServiceFailure("REFUND_NOT_FOUND", "This proposal has no refund operation.")
    if operation.status in FINAL:
        return operation
    proposal = load_proposal(db, proposal_id)
    try:
        found = payments.get("/refunds", params={"idempotency_key": operation.idempotency_key})
    except ServiceFailure:
        return operation  # still uncertain; never assume either way
    if not found:
        if operation.status == "submitted":
            # The provider accepted it before; losing it is for a person to look at.
            return operation
        return send(db, payments, proposal, operation)

    refund = found[0]
    status = FROM_PROVIDER.get(refund["status"], "unknown")
    if (status, refund["provider_reference"]) == (operation.status, operation.provider_reference):
        return operation
    return record_outcome(
        db,
        proposal,
        operation,
        status=status,
        reference=refund["provider_reference"],
        response=refund,
        event_type="refund_reconciled",
        details={"previous_status": operation.status},
    )


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
        status = FROM_PROVIDER.get(body.get("status"), "failed")
        reference = body.get("provider_reference")
        response = body
    return record_outcome(
        db,
        proposal,
        operation,
        status=status,
        reference=reference,
        response=response,
        event_type="refund_executed",
        details={"error_code": response.get("error_code")},
    )


def record_outcome(
    db: Session,
    proposal,
    operation: RefundOperationView,
    *,
    status: str,
    reference: str | None,
    response: dict[str, Any],
    event_type: str,
    details: dict[str, Any],
) -> RefundOperationView:
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
        event_type=event_type,
        payload={
            "operation_id": operation.operation_id,
            "proposal_id": operation.proposal_id,
            "status": status,
            "amount_minor": operation.amount_minor,
            **details,
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


def load_proposal(db: Session, proposal_id: str):
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
    return proposal
