"""Deterministic refund eligibility (W1.14). Ordinary code decides, never the model (W2.12).

`decide_refund_eligibility` is a pure function of the facts: no database, network or clock
(today comes in as `as_of`). `check_refund_eligibility` gathers the facts for an order.
The decision is one of four outcomes:

- eligible:           a refund proposal may be created, for at most `max_refund_minor`
- already_processing: a refund for the order is in flight; tell the customer, create nothing
- blocked:            the policy says no (for now); answer the customer with the reason
- needs_review:       the policy refers the case to a human reviewer

The first rule that fails decides the outcome, in the order below. All amounts are integer
minor units (paise).
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.auth.reviewer import required_reviewer_role
from app.db import SCHEMA
from app.schemas.payments import PaymentStatusView
from app.services.orders import ensure_order_owner
from app.services.providers import ProviderClient, parse

# Dates are compared as calendar dates in the store's time zone.
STORE_TZ = timezone(timedelta(hours=5, minutes=30))


@dataclass(frozen=True)
class RefundPolicyVersion:
    version: str
    effective_from: date
    effective_to: date | None
    # A return must be requested within this many days of delivery (section 3).
    return_window_days: int

    def applies_on(self, day: date) -> bool:
        return self.effective_from <= day and (
            self.effective_to is None or day <= self.effective_to
        )


# From app/policies/documents/refund_policy_v*.md; a test checks the two stay in sync.
REFUND_POLICY_VERSIONS = (
    RefundPolicyVersion("v0", date(2025, 1, 1), date(2025, 12, 31), 7),
    RefundPolicyVersion("v1", date(2026, 1, 1), None, 10),
)
LATEST = REFUND_POLICY_VERSIONS[-1]

# Local refund operation statuses that hold part of the refundable balance.
HELD_STATUSES = {"reserved", "submitted", "succeeded", "unknown"}
# ...and the ones that mean a refund is still in flight.
IN_FLIGHT_STATUSES = {"reserved", "submitted", "unknown"}


@dataclass(frozen=True)
class RefundRecord:
    status: str
    amount_minor: int


@dataclass(frozen=True)
class RefundFacts:
    """Everything the decision needs, gathered by `check_refund_eligibility`."""

    order_id: str
    # Today in the store's time zone: a return not requested yet would be requested today.
    as_of: date
    delivered_at: datetime | None
    paid_amount_minor: int
    # Sum of the order lines: the price paid for the items, without shipping (section 4).
    items_total_minor: int
    return_status: str | None = None
    return_requested_at: datetime | None = None
    return_received_at: datetime | None = None
    inspection_status: str | None = None
    # refund_operations rows for the order: reserved | submitted | succeeded | failed | unknown
    local_refunds: list[RefundRecord] = field(default_factory=list)
    # The payment provider's view (Q2): its refundable balance and refunds.
    provider_refundable_minor: int = 0
    provider_refunds: list[RefundRecord] = field(default_factory=list)
    # The amount asked for; None means "as much as is allowed".
    requested_amount_minor: int | None = None


@dataclass(frozen=True)
class EligibilityDecision:
    order_id: str
    # eligible | already_processing | blocked | needs_review
    outcome: str
    # Why it is not eligible, e.g. RETURN_NOT_RECEIVED; None when eligible.
    reason: str | None
    # Citations such as refund_policy_v1:section_3.
    policy_refs: list[str]
    # The refund policy version in effect on the delivery date, if any.
    policy_version: str | None
    # The smaller of the local and provider balances.
    refundable_balance_minor: int
    # The most that may be refunded: the items total, capped by the refundable balance.
    max_refund_minor: int
    # When eligible: the amount to propose, and who must approve it.
    amount_minor: int | None = None
    reviewer_role: str | None = None


def policy_for_delivery(delivered_on: date) -> RefundPolicyVersion | None:
    for version in REFUND_POLICY_VERSIONS:
        if version.applies_on(delivered_on):
            return version
    return None


def store_date(moment: datetime) -> date:
    return moment.astimezone(STORE_TZ).date()


def refundable_balance(facts: RefundFacts) -> int:
    """Paid amount minus successful and pending refunds, by whichever ledger shows less."""
    held_locally = sum(r.amount_minor for r in facts.local_refunds if r.status in HELD_STATUSES)
    local = facts.paid_amount_minor - held_locally
    return max(0, min(local, facts.provider_refundable_minor))


def decide_refund_eligibility(facts: RefundFacts) -> EligibilityDecision:
    balance = refundable_balance(facts)
    max_refund = min(facts.items_total_minor, balance)

    def decision(
        outcome: str, reason: str | None, refs: list[str], version: str | None, **extra: object
    ) -> EligibilityDecision:
        return EligibilityDecision(
            order_id=facts.order_id,
            outcome=outcome,
            reason=reason,
            policy_refs=refs,
            policy_version=version,
            refundable_balance_minor=balance,
            max_refund_minor=max_refund,
            **extra,
        )

    # 1. Only a delivered order can be returned and refunded.
    if facts.delivered_at is None:
        return decision(
            "blocked", "ORDER_NOT_DELIVERED", [f"refund_policy_{LATEST.version}:section_3"], None
        )

    # 2. The policy in effect on the delivery date decides; with none, a human does.
    policy = policy_for_delivery(store_date(facts.delivered_at))
    if policy is None:
        return decision(
            "needs_review", "NO_POLICY_IN_EFFECT", ["human_review_exceptions_v1:section_4"], None
        )
    v = policy.version
    refund = f"refund_policy_{v}"

    # 3. A refund already in flight: tell the customer it is being processed (section 7).
    in_flight = any(r.status in IN_FLIGHT_STATUSES for r in facts.local_refunds) or any(
        r.status == "pending" for r in facts.provider_refunds
    )
    if in_flight:
        return decision("already_processing", "REFUND_PENDING", [f"{refund}:section_7"], v)

    # 4. Nothing left to refund after earlier refunds (section 7).
    refunded_before = any(r.status == "succeeded" for r in facts.local_refunds) or any(
        r.status == "succeeded" for r in facts.provider_refunds
    )
    if balance == 0 and refunded_before:
        return decision("blocked", "ALREADY_REFUNDED", [f"{refund}:section_7"], v)

    # 5. A return must have been requested, within the window (section 3). With no return
    # yet, a request made today would be the one counted.
    no_return = facts.return_status is None or facts.return_requested_at is None
    requested_on = facts.as_of if no_return else store_date(facts.return_requested_at)
    days_after_delivery = (requested_on - store_date(facts.delivered_at)).days
    if days_after_delivery > policy.return_window_days:
        return decision("needs_review", "OUTSIDE_RETURN_WINDOW", [f"{refund}:section_3"], v)
    if no_return:
        return decision(
            "blocked",
            "RETURN_NOT_REQUESTED",
            [f"{refund}:section_3", "return_processing_v1:section_2"],
            v,
        )

    # 6. Received means a received status and date; a tracking update is not enough.
    if facts.return_status not in ("received", "closed") or facts.return_received_at is None:
        return decision(
            "blocked",
            "RETURN_NOT_RECEIVED",
            [f"{refund}:section_7", "return_processing_v1:section_3"],
            v,
        )

    # 7. Inspection must have passed.
    if facts.inspection_status == "failed":
        return decision("blocked", "INSPECTION_FAILED", ["return_processing_v1:section_5"], v)
    if facts.inspection_status != "passed":
        return decision(
            "blocked",
            "INSPECTION_PENDING",
            [f"{refund}:section_7", "return_processing_v1:section_4"],
            v,
        )

    # 8. The amount: positive, and no more than the items total and the balance (section 4).
    amount = (
        facts.requested_amount_minor if facts.requested_amount_minor is not None else max_refund
    )
    if amount <= 0:
        reason = (
            "INVALID_AMOUNT" if facts.requested_amount_minor is not None else "NOTHING_TO_REFUND"
        )
        return decision("blocked", reason, [f"{refund}:section_4"], v)
    if amount > max_refund:
        return decision("blocked", "AMOUNT_EXCEEDS_REFUNDABLE", [f"{refund}:section_4"], v)

    role = required_reviewer_role(amount)
    refs = [f"{refund}:section_3", f"{refund}:section_4", f"{refund}:section_5"]
    if role == "senior_reviewer":
        refs.append("human_review_exceptions_v1:section_3")
    return decision("eligible", None, refs, v, amount_minor=amount, reviewer_role=role)


def check_refund_eligibility(
    db: Session,
    payments: ProviderClient,
    customer: AuthenticatedCustomer,
    order_id: str,
    *,
    amount_minor: int | None = None,
    today: date | None = None,
) -> EligibilityDecision:
    """Gather the facts for one of the customer's orders and decide.

    Raises ORDER_NOT_FOUND for a missing or someone else's order, and a retryable provider
    failure when the payment provider cannot be asked: no decision is made without it.
    """
    ensure_order_owner(db, customer, order_id)
    order = db.execute(
        text(
            "SELECT o.delivered_at, o.paid_amount_minor, "
            "(SELECT COALESCE(SUM(i.paid_amount_minor), 0) "
            f" FROM {SCHEMA}.order_items i WHERE i.order_id = o.id) AS items_total_minor, "
            "r.status AS return_status, r.requested_at, r.received_at, r.inspection_status "
            f"FROM {SCHEMA}.orders o LEFT JOIN {SCHEMA}.returns r ON r.order_id = o.id "
            "WHERE o.id = :order"
        ),
        {"order": order_id},
    ).one()
    local_refunds = db.execute(
        text(
            f"SELECT status, amount_minor FROM {SCHEMA}.refund_operations WHERE order_id = :order"
        ),
        {"order": order_id},
    ).all()
    payment = parse(PaymentStatusView, payments.get(f"/payments/{order_id}"), "payment")

    facts = RefundFacts(
        order_id=order_id,
        as_of=today or store_date(datetime.now(STORE_TZ)),
        delivered_at=order.delivered_at,
        paid_amount_minor=order.paid_amount_minor,
        items_total_minor=order.items_total_minor,
        return_status=order.return_status,
        return_requested_at=order.requested_at,
        return_received_at=order.received_at,
        inspection_status=order.inspection_status,
        local_refunds=[RefundRecord(r.status, r.amount_minor) for r in local_refunds],
        provider_refundable_minor=payment.refundable_minor,
        provider_refunds=[RefundRecord(r.status, r.amount_minor) for r in payment.refunds],
        requested_amount_minor=amount_minor,
    )
    return decide_refund_eligibility(facts)
