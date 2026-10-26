"""Deterministic refund eligibility: every rule, its boundaries and its priority (W1.14)."""

import re
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.services.eligibility import (
    REFUND_POLICY_VERSIONS,
    RefundFacts,
    RefundRecord,
    decide_refund_eligibility,
    policy_for_delivery,
    refundable_balance,
)

IST = timezone(timedelta(hours=5, minutes=30))
POLICIES = Path(__file__).resolve().parents[2] / "app/policies/documents"


def ist(year: int, month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=IST)


# The reference case: headphones, returned, received and passed inspection (like ord_1001).
ELIGIBLE = RefundFacts(
    order_id="ord_1001",
    as_of=date(2026, 10, 25),
    delivered_at=ist(2026, 10, 16),
    paid_amount_minor=249_900,
    items_total_minor=249_900,
    return_status="received",
    return_requested_at=ist(2026, 10, 18),
    return_received_at=ist(2026, 10, 22),
    inspection_status="passed",
    provider_refundable_minor=249_900,
)
NO_RETURN = {
    "return_status": None,
    "return_requested_at": None,
    "return_received_at": None,
    "inspection_status": None,
}


def decide(**changes: object):
    return decide_refund_eligibility(replace(ELIGIBLE, **changes))


def outcome(**changes: object) -> tuple[str, str | None]:
    result = decide(**changes)
    return result.outcome, result.reason


# --- Eligible ---------------------------------------------------------------------------


def test_reference_case_is_eligible_for_the_full_amount() -> None:
    result = decide()
    assert (result.outcome, result.reason) == ("eligible", None)
    assert (result.amount_minor, result.max_refund_minor, result.refundable_balance_minor) == (
        249_900,
        249_900,
        249_900,
    )
    assert (result.policy_version, result.reviewer_role) == ("v1", "reviewer")
    assert result.policy_refs == [
        "refund_policy_v1:section_3",
        "refund_policy_v1:section_4",
        "refund_policy_v1:section_5",
    ]


def test_a_smaller_amount_can_be_requested() -> None:
    result = decide(requested_amount_minor=100_000)
    assert (result.outcome, result.amount_minor) == ("eligible", 100_000)


def test_closed_return_that_passed_inspection_is_still_eligible() -> None:
    assert outcome(return_status="closed") == ("eligible", None)


def test_high_value_refund_needs_a_senior_reviewer() -> None:
    result = decide(
        paid_amount_minor=8_999_900,
        items_total_minor=8_999_900,
        provider_refundable_minor=8_999_900,
    )
    assert (result.outcome, result.reviewer_role) == ("eligible", "senior_reviewer")
    assert result.policy_refs[-1] == "human_review_exceptions_v1:section_3"


def test_exactly_50000_rupees_needs_a_standard_reviewer() -> None:
    result = decide(
        paid_amount_minor=5_000_000,
        items_total_minor=5_000_000,
        provider_refundable_minor=5_000_000,
    )
    assert result.reviewer_role == "reviewer"


def test_an_earlier_failed_refund_does_not_block() -> None:
    result = decide(
        local_refunds=[RefundRecord("failed", 249_900)],
        provider_refunds=[RefundRecord("failed", 249_900)],
    )
    assert (result.outcome, result.amount_minor) == ("eligible", 249_900)


# --- Amounts (refund_policy section 4) --------------------------------------------------


def test_shipping_is_not_refunded() -> None:
    result = decide(paid_amount_minor=259_800, provider_refundable_minor=259_800)
    assert (result.outcome, result.amount_minor, result.refundable_balance_minor) == (
        "eligible",
        249_900,
        259_800,
    )
    assert outcome(
        paid_amount_minor=259_800, provider_refundable_minor=259_800, requested_amount_minor=249_901
    ) == ("blocked", "AMOUNT_EXCEEDS_REFUNDABLE")


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (249_900, ("eligible", None)),
        (249_901, ("blocked", "AMOUNT_EXCEEDS_REFUNDABLE")),
        (1, ("eligible", None)),
        (0, ("blocked", "INVALID_AMOUNT")),
        (-100, ("blocked", "INVALID_AMOUNT")),
    ],
)
def test_requested_amount_boundaries(requested: int, expected: tuple[str, str | None]) -> None:
    assert outcome(requested_amount_minor=requested) == expected


def test_partial_earlier_refund_leaves_the_rest() -> None:
    earlier = [RefundRecord("succeeded", 100_000)]
    result = decide(
        local_refunds=earlier, provider_refunds=earlier, provider_refundable_minor=149_900
    )
    assert (result.outcome, result.amount_minor, result.max_refund_minor) == (
        "eligible",
        149_900,
        149_900,
    )


# --- Refundable balance: the smaller of the two ledgers ---------------------------------


def test_balance_uses_the_local_ledger_when_it_shows_less() -> None:
    # A reserved operation the provider has not seen yet still holds the amount.
    facts = replace(ELIGIBLE, local_refunds=[RefundRecord("reserved", 200_000)])
    assert refundable_balance(facts) == 49_900


def test_balance_uses_the_provider_when_it_shows_less() -> None:
    # A refund made directly at the provider, with no local record.
    facts = replace(ELIGIBLE, provider_refundable_minor=49_900)
    assert refundable_balance(facts) == 49_900
    assert decide(provider_refundable_minor=49_900).amount_minor == 49_900


@pytest.mark.parametrize(
    ("status", "held"),
    [
        ("reserved", True),
        ("submitted", True),
        ("succeeded", True),
        ("unknown", True),
        ("failed", False),
    ],
)
def test_which_local_statuses_hold_the_balance(status: str, held: bool) -> None:
    facts = replace(ELIGIBLE, local_refunds=[RefundRecord(status, 100_000)])
    assert refundable_balance(facts) == (149_900 if held else 249_900)


def test_balance_never_goes_negative() -> None:
    facts = replace(ELIGIBLE, local_refunds=[RefundRecord("succeeded", 300_000)])
    assert refundable_balance(facts) == 0


# --- Refund already in flight or done (refund_policy section 7) -------------------------


@pytest.mark.parametrize("status", ["reserved", "submitted", "unknown"])
def test_local_refund_in_flight(status: str) -> None:
    result = decide(local_refunds=[RefundRecord(status, 249_900)])
    assert (result.outcome, result.reason) == ("already_processing", "REFUND_PENDING")
    assert result.policy_refs == ["refund_policy_v1:section_7"]
    assert result.amount_minor is None


def test_provider_refund_pending() -> None:
    assert outcome(
        provider_refunds=[RefundRecord("pending", 249_900)], provider_refundable_minor=0
    ) == (
        "already_processing",
        "REFUND_PENDING",
    )


def test_fully_refunded() -> None:
    done = [RefundRecord("succeeded", 249_900)]
    assert outcome(local_refunds=done, provider_refunds=done, provider_refundable_minor=0) == (
        "blocked",
        "ALREADY_REFUNDED",
    )


def test_refund_seen_only_at_the_provider_counts() -> None:
    assert outcome(
        provider_refunds=[RefundRecord("succeeded", 249_900)], provider_refundable_minor=0
    ) == (
        "blocked",
        "ALREADY_REFUNDED",
    )


# --- Return and inspection --------------------------------------------------------------


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 10, 26), ("blocked", "RETURN_NOT_REQUESTED")),  # day 10: still in time
        (date(2026, 10, 27), ("needs_review", "OUTSIDE_RETURN_WINDOW")),  # day 11: too late
    ],
)
def test_no_return_yet_is_judged_as_if_requested_today(
    today: date, expected: tuple[str, str]
) -> None:
    assert outcome(as_of=today, **NO_RETURN) == expected


def test_no_return_requested() -> None:
    result = decide(**NO_RETURN)
    assert (result.outcome, result.reason) == ("blocked", "RETURN_NOT_REQUESTED")
    assert result.policy_refs == ["refund_policy_v1:section_3", "return_processing_v1:section_2"]


@pytest.mark.parametrize("status", ["requested", "in_transit"])
def test_return_not_received(status: str) -> None:
    result = decide(return_status=status, return_received_at=None, inspection_status=None)
    assert (result.outcome, result.reason) == ("blocked", "RETURN_NOT_RECEIVED")
    assert "return_processing_v1:section_3" in result.policy_refs


def test_received_status_without_a_date_is_not_received() -> None:
    assert outcome(return_received_at=None) == ("blocked", "RETURN_NOT_RECEIVED")


@pytest.mark.parametrize(
    ("inspection", "reason"),
    [
        ("pending", "INSPECTION_PENDING"),
        (None, "INSPECTION_PENDING"),
        ("failed", "INSPECTION_FAILED"),
    ],
)
def test_inspection_must_pass(inspection: str | None, reason: str) -> None:
    assert outcome(inspection_status=inspection) == ("blocked", reason)


def test_failed_inspection_cites_return_processing() -> None:
    assert decide(inspection_status="failed").policy_refs == ["return_processing_v1:section_5"]


def test_order_not_delivered() -> None:
    result = decide(delivered_at=None, **NO_RETURN)
    assert (result.outcome, result.reason, result.policy_version) == (
        "blocked",
        "ORDER_NOT_DELIVERED",
        None,
    )


# --- Return window and policy version (refund_policy section 3) -------------------------


@pytest.mark.parametrize(
    ("days", "expected"),
    [(0, "eligible"), (10, "eligible"), (11, "needs_review")],
)
def test_v1_window_is_10_days(days: int, expected: str) -> None:
    result = decide(return_requested_at=ELIGIBLE.delivered_at + timedelta(days=days))
    assert result.outcome == expected
    if expected == "needs_review":
        assert (result.reason, result.policy_refs) == (
            "OUTSIDE_RETURN_WINDOW",
            ["refund_policy_v1:section_3"],
        )


@pytest.mark.parametrize(("days", "expected"), [(7, "eligible"), (8, "needs_review")])
def test_v0_window_is_7_days(days: int, expected: str) -> None:
    delivered = ist(2025, 6, 10)
    result = decide(
        delivered_at=delivered,
        return_requested_at=delivered + timedelta(days=days),
        return_received_at=delivered + timedelta(days=days + 4),
    )
    assert (result.outcome, result.policy_version) == (expected, "v0")


def test_version_is_chosen_by_delivery_date() -> None:
    # Delivered on the last day of v0, requested 8 days later in 2026: v0's 7 days apply.
    delivered = ist(2025, 12, 31)
    result = decide(
        delivered_at=delivered,
        return_requested_at=delivered + timedelta(days=8),
        return_received_at=delivered + timedelta(days=12),
    )
    assert (result.outcome, result.policy_version) == ("needs_review", "v0")


def test_window_counts_calendar_days_in_india() -> None:
    # 00:30 IST on 17 Oct is still 16 Oct in UTC. Counted in IST, 27 Oct is day 10.
    delivered = ist(2026, 10, 17, 0, 30).astimezone(UTC)
    requested = ist(2026, 10, 27, 10).astimezone(UTC)
    assert outcome(delivered_at=delivered, return_requested_at=requested) == ("eligible", None)


def test_no_policy_in_effect_before_2025() -> None:
    result = decide(delivered_at=ist(2024, 12, 31), return_requested_at=ist(2025, 1, 2))
    assert (result.outcome, result.reason, result.policy_refs) == (
        "needs_review",
        "NO_POLICY_IN_EFFECT",
        ["human_review_exceptions_v1:section_4"],
    )


@pytest.mark.parametrize(
    ("day", "version"),
    [
        (date(2025, 1, 1), "v0"),
        (date(2025, 12, 31), "v0"),
        (date(2026, 1, 1), "v1"),
        (date(2030, 6, 1), "v1"),
        (date(2024, 12, 31), None),
    ],
)
def test_policy_for_delivery(day: date, version: str | None) -> None:
    found = policy_for_delivery(day)
    assert (found.version if found else None) == version


# --- Rule priority ----------------------------------------------------------------------


def test_refund_in_flight_wins_over_a_failed_inspection() -> None:
    assert outcome(inspection_status="failed", local_refunds=[RefundRecord("submitted", 249_900)])[
        0
    ] == ("already_processing")


def test_window_wins_over_a_return_not_yet_received() -> None:
    assert outcome(
        return_requested_at=ist(2026, 10, 30),
        return_status="in_transit",
        return_received_at=None,
        inspection_status=None,
    ) == ("needs_review", "OUTSIDE_RETURN_WINDOW")


def test_already_refunded_wins_over_a_missing_return() -> None:
    done = [RefundRecord("succeeded", 249_900)]
    assert outcome(
        local_refunds=done, provider_refunds=done, provider_refundable_minor=0, **NO_RETURN
    ) == (
        "blocked",
        "ALREADY_REFUNDED",
    )


def test_missing_policy_wins_over_everything_after_delivery() -> None:
    assert (
        outcome(delivered_at=ist(2024, 6, 1), inspection_status="failed")[1]
        == "NO_POLICY_IN_EFFECT"
    )


# --- The code matches the policy documents ----------------------------------------------


def front_matter(text: str) -> dict[str, str]:
    block = text.split("---")[1]
    return dict(line.split(": ", 1) for line in block.strip().splitlines())


@pytest.mark.parametrize("version", REFUND_POLICY_VERSIONS, ids=lambda v: v.version)
def test_policy_table_matches_the_documents(version) -> None:
    text = (POLICIES / f"refund_policy_{version.version}.md").read_text(encoding="utf-8")
    meta = front_matter(text)
    assert meta["version"] == version.version
    assert date.fromisoformat(meta["effective_from"]) == version.effective_from
    effective_to = (
        None if meta["effective_to"] == "null" else date.fromisoformat(meta["effective_to"])
    )
    assert effective_to == version.effective_to
    window = re.search(r"within (\d+) days of the delivery date", text)
    assert window and int(window.group(1)) == version.return_window_days


def test_every_refund_policy_document_is_in_the_table() -> None:
    documents = {p.stem.removeprefix("refund_policy_") for p in POLICIES.glob("refund_policy_*.md")}
    assert documents == {v.version for v in REFUND_POLICY_VERSIONS}
