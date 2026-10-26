"""Typed results of the payment and refund read services (W1.12)."""

from datetime import datetime

from pydantic import BaseModel


class ProviderRefund(BaseModel):
    provider_reference: str
    amount_minor: int
    # pending | succeeded | failed
    status: str
    created_at: datetime


class PaymentStatusView(BaseModel):
    """The payment as the payment provider reports it; the provider is the source of truth (Q2)."""

    order_id: str
    payment_id: str
    status: str
    currency: str
    amount_minor: int
    # Paid amount minus succeeded and pending refunds, as the provider counts it.
    refundable_minor: int
    refunds: list[ProviderRefund]


class RefundStatusView(BaseModel):
    """A refund operation as recorded locally, checked against the payment provider.

    Only a provider-confirmed status may be told to the customer (W3.14). When the provider
    cannot be reached, `provider_status` is None and `provider_error` says why.
    """

    operation_id: str
    order_id: str
    amount_minor: int
    # reserved | submitted | succeeded | failed | unknown
    local_status: str
    provider_reference: str | None
    # pending | succeeded | failed, "no_record" if the provider has none, None if unreachable
    provider_status: str | None
    provider_error: str | None
    # Whether the two agree; None when the provider could not be asked.
    matches: bool | None
    created_at: datetime
    updated_at: datetime
