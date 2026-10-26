"""Payment and refund read services (W1.12).

The payment provider is the source of truth for payments (Q2). Refund operations are recorded
locally and checked against the provider by idempotency key before their status is trusted.
"""

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.schemas.payments import PaymentStatusView, RefundStatusView
from app.services.errors import ServiceFailure
from app.services.orders import ensure_order_owner
from app.services.providers import ProviderClient, parse

# The provider status each local status should agree with ("no_record": not sent, or rejected).
EXPECTED_PROVIDER_STATUS = {
    "reserved": {"no_record"},
    "submitted": {"pending"},
    "succeeded": {"succeeded"},
    "failed": {"failed", "no_record"},
    # The outcome is not known locally, so it never agrees until reconciled (W3.12).
    "unknown": set(),
}


def get_payment_status(
    db: Session, payments: ProviderClient, customer: AuthenticatedCustomer, order_id: str
) -> PaymentStatusView:
    ensure_order_owner(db, customer, order_id)
    return parse(PaymentStatusView, payments.get(f"/payments/{order_id}"), "payment")


def get_refund_status(
    db: Session, payments: ProviderClient, customer: AuthenticatedCustomer, operation_id: str
) -> RefundStatusView:
    operation = db.execute(
        text(
            "SELECT r.id, r.order_id, r.amount_minor, r.status, r.idempotency_key, "
            "r.provider_reference, r.created_at, r.updated_at "
            f"FROM {SCHEMA}.refund_operations r JOIN {SCHEMA}.orders o ON o.id = r.order_id "
            "WHERE r.id = :operation AND o.customer_id = :customer"
        ),
        {"operation": operation_id, "customer": customer.customer_id},
    ).one_or_none()
    if operation is None:
        # Same failure for a missing operation and someone else's.
        raise ServiceFailure("REFUND_NOT_FOUND", "No refund with this ID was found.")

    provider_status: str | None = None
    provider_reference = operation.provider_reference
    provider_error: str | None = None
    try:
        found = payments.get("/refunds", params={"idempotency_key": operation.idempotency_key})
    except ServiceFailure as failure:
        provider_error = failure.error_code
    else:
        if found:
            provider_status = found[0]["status"]
            provider_reference = provider_reference or found[0]["provider_reference"]
        else:
            provider_status = "no_record"

    return RefundStatusView(
        operation_id=operation.id,
        order_id=operation.order_id,
        amount_minor=operation.amount_minor,
        local_status=operation.status,
        provider_reference=provider_reference,
        provider_status=provider_status,
        provider_error=provider_error,
        matches=(
            None
            if provider_status is None
            else provider_status in EXPECTED_PROVIDER_STATUS[operation.status]
        ),
        created_at=operation.created_at,
        updated_at=operation.updated_at,
    )
