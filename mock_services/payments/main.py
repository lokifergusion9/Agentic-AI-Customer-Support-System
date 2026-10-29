"""Mock payment / refund API (W1.8): `uvicorn mock_services.payments.main:app --port 8001`.

Every endpoint except `/health` needs the `X-API-Key` header (see `mock_services.common`).

Idempotency contract (PDF section 12): the same key with the same payload returns the same
refund; the same key with a changed payload is rejected.
"""

import secrets
import time
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, Query, Response, status
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mock_services.common import (
    FaultRequestBase,
    MockError,
    add_error_handler,
    add_fault_routes,
    api_key_check,
    raise_fault_error,
)
from mock_services.payments.store import Payment, Refund, store


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Local-only default matching .env.example.
    mock_payments_api_key: str = "mock-payments-local-key"


settings = Settings()


app = FastAPI(title="Mock payment provider")
add_error_handler(app)
api = APIRouter(dependencies=[Depends(api_key_check(settings.mock_payments_api_key))])


class RefundOut(BaseModel):
    provider_reference: str
    idempotency_key: str
    order_id: str
    amount_minor: int
    currency: str
    status: str
    created_at: datetime


class PaymentOut(BaseModel):
    payment_id: str
    order_id: str
    amount_minor: int
    currency: str
    status: str
    captured_at: datetime
    refundable_minor: int
    refunds: list[RefundOut]


def refund_out(refund: Refund) -> RefundOut:
    return RefundOut(**refund.__dict__)


def payment_out(payment: Payment) -> PaymentOut:
    return PaymentOut(
        payment_id=payment.payment_id,
        order_id=payment.order_id,
        amount_minor=payment.amount_minor,
        currency=payment.currency,
        status=payment.status,
        captured_at=payment.captured_at,
        refundable_minor=payment.refundable_minor,
        refunds=[refund_out(r) for r in payment.refunds],
    )


# --- Admin ------------------------------------------------------------------------------


class SeedRefund(BaseModel):
    provider_reference: str
    idempotency_key: str
    amount_minor: int
    status: str


class SeedPayment(BaseModel):
    payment_id: str
    order_id: str
    amount_minor: int
    currency: str
    status: str
    captured_at: datetime
    refunds: list[SeedRefund] = []


class ResetRequest(BaseModel):
    payments: list[SeedPayment]


@api.post("/admin/reset")
def reset(request: ResetRequest) -> dict[str, int]:
    """Replace all state with the seed data and clear queued faults."""
    payments = [
        Payment(
            payment_id=p.payment_id,
            order_id=p.order_id,
            amount_minor=p.amount_minor,
            currency=p.currency,
            status=p.status,
            captured_at=p.captured_at,
            refunds=[
                Refund(
                    provider_reference=r.provider_reference,
                    idempotency_key=r.idempotency_key,
                    order_id=p.order_id,
                    amount_minor=r.amount_minor,
                    currency=p.currency,
                    status=r.status,
                    created_at=p.captured_at,
                )
                for r in p.refunds
            ],
        )
        for p in request.payments
    ]
    store.reset(payments)
    return {"payments": len(payments), "refunds": sum(len(p.refunds) for p in payments)}


@app.get("/health")
def health() -> dict[str, str]:
    """Public, for the docker healthcheck."""
    return {"status": "ok"}


# --- Queries ----------------------------------------------------------------------------


@api.get("/payments/{order_id}")
def get_payment(order_id: str) -> PaymentOut:
    fault = store.faults.take("payment.get", queued=False, order_id=order_id)
    if fault is not None:
        if fault.mode == "error":
            raise_fault_error(fault)
        time.sleep(fault.delay_seconds)  # a read has nothing to accept: every delay mode holds
    with store.lock:
        payment = store.payments.get(order_id)
        if payment is None:
            raise MockError(
                status.HTTP_404_NOT_FOUND, "PAYMENT_NOT_FOUND", "No payment for this order."
            )
        return payment_out(payment)


@api.get("/refunds")
def find_refunds(
    order_id: Annotated[str | None, Query()] = None,
    idempotency_key: Annotated[str | None, Query()] = None,
) -> list[RefundOut]:
    """Refunds for an order, or the refund for an idempotency key (used to reconcile)."""
    if (order_id is None) == (idempotency_key is None):
        raise MockError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "INVALID_QUERY",
            "Pass exactly one of order_id or idempotency_key.",
        )
    with store.lock:
        if idempotency_key is not None:
            refund = store.by_key.get(idempotency_key)
            return [refund_out(refund)] if refund else []
        return [refund_out(r) for r in store.refunds.values() if r.order_id == order_id]


@api.get("/refunds/{provider_reference}")
def get_refund(provider_reference: str) -> RefundOut:
    with store.lock:
        refund = store.refunds.get(provider_reference)
        if refund is None:
            raise MockError(status.HTTP_404_NOT_FOUND, "REFUND_NOT_FOUND", "No such refund.")
        return refund_out(refund)


# --- Refunds ----------------------------------------------------------------------------


class RefundRequest(BaseModel):
    idempotency_key: str = Field(min_length=1, max_length=200)
    order_id: str = Field(min_length=1)
    amount_minor: int = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)


def same_payload(refund: Refund, request: RefundRequest) -> bool:
    return (refund.order_id, refund.amount_minor, refund.currency) == (
        request.order_id,
        request.amount_minor,
        request.currency,
    )


@api.post(
    "/refunds",
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_200_OK: {"description": "Replay of an earlier request"}},
)
def create_refund(request: RefundRequest, response: Response) -> RefundOut:
    # The whole check-and-insert runs under the lock, so two concurrent requests with the
    # same key cannot both create a refund.
    with store.lock:
        existing = store.by_key.get(request.idempotency_key)
        if existing is not None:
            if not same_payload(existing, request):
                raise MockError(
                    status.HTTP_409_CONFLICT,
                    "IDEMPOTENCY_KEY_REUSED",
                    "This idempotency key was already used with a different payload.",
                )
            response.status_code = status.HTTP_200_OK
            return refund_out(existing)

        payment = store.payments.get(request.order_id)
        if payment is None:
            raise MockError(
                status.HTTP_404_NOT_FOUND, "PAYMENT_NOT_FOUND", "No payment for this order."
            )
        if request.currency != payment.currency:
            raise MockError(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "CURRENCY_MISMATCH",
                f"The payment was made in {payment.currency}.",
            )
        if request.amount_minor > payment.refundable_minor:
            raise MockError(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "AMOUNT_EXCEEDS_REFUNDABLE",
                f"At most {payment.refundable_minor} minor units can be refunded.",
            )

        # Faults apply only to new, valid refunds; a replay above always returns the stored one.
        fault = store.faults.take("refund.create", queued=True, order_id=request.order_id)
        if fault is not None and fault.mode == "error":
            raise_fault_error(fault)

        refund = Refund(
            provider_reference=f"rf_{secrets.token_hex(8)}",
            idempotency_key=request.idempotency_key,
            order_id=request.order_id,
            amount_minor=request.amount_minor,
            currency=request.currency,
            status="pending" if fault is not None and fault.mode == "delay" else "succeeded",
            created_at=datetime.now(UTC),
        )
        store.add(payment, refund)
        result = refund_out(refund)

    if fault is not None and fault.mode == "accept_then_timeout":
        # The refund is already recorded; outlive the client's timeout so it never learns
        # the result. Sleeping outside the lock keeps the reconciliation queries working.
        time.sleep(fault.delay_seconds)
    return result


# --- Fault injection --------------------------------------------------------------------


# Logical operations a keyed fault can target, matched by order_id.
OPERATIONS = frozenset({"payment.get", "refund.create"})


class FaultRequest(FaultRequestBase):
    """Applied to the next `count` new refund requests, in order.

    error: the refund is not recorded. delay: the refund stays pending until settled.
    accept_then_timeout: the refund is recorded, then the response is held `delay_seconds`.
    """

    mode: Literal["error", "delay", "accept_then_timeout"]


add_fault_routes(api, store.faults, FaultRequest, OPERATIONS)


class SettleRequest(BaseModel):
    status: Literal["succeeded", "failed"]


@api.post("/admin/refunds/{provider_reference}/settle")
def settle_refund(provider_reference: str, request: SettleRequest) -> RefundOut:
    """Finish a refund left pending by a `delay` fault. A failed refund frees its amount."""
    with store.lock:
        refund = store.refunds.get(provider_reference)
        if refund is None:
            raise MockError(status.HTTP_404_NOT_FOUND, "REFUND_NOT_FOUND", "No such refund.")
        if refund.status != "pending":
            raise MockError(
                status.HTTP_409_CONFLICT,
                "REFUND_NOT_PENDING",
                f"The refund is already {refund.status}.",
            )
        refund.status = request.status
        return refund_out(refund)


app.include_router(api)
