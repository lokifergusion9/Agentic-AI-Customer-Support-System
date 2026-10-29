"""Outcome evaluator: the business state the attempt left behind (W4.9, E§13).

Read from the snapshot (the database and the mock providers' records), never from the final
message. Refunds the fixture started with are not counted.
"""

from evals.evaluators.base import AttemptRecord, Score, Scorer

NAME, VERSION = "outcome", "1.0.0"
score = Scorer(NAME, VERSION)


def evaluate(record: AttemptRecord) -> list[Score]:
    return [
        refund_amount(record),
        refund_count(record),
        refund_order(record),
        approval_requested(record),
        ticket_status(record),
        run_status(record),
    ]


def refund_amount(record: AttemptRecord) -> Score:
    expected = record.task.evaluation.outcome.refund_total_minor
    observed = sum(r["amount_minor"] for r in record.confirmed_refunds)
    evidence = {"expected_amount_minor": expected, "observed_amount_minor": observed}
    if observed == expected:
        return score.passed("amount", **evidence)
    if expected == 0:
        return score.failed("amount", "UNEXPECTED_REFUND", **evidence)
    if observed == 0:
        return score.failed("amount", "MISSING_REFUND", **evidence)
    return score.failed("amount", "WRONG_REFUND_AMOUNT", **evidence)


def refund_count(record: AttemptRecord) -> Score:
    """Exactly the expected number of financial effects at the provider, pending or not."""
    expected = record.task.evaluation.outcome.refund_count
    observed = len(record.new_refunds)
    evidence = {"expected_count": expected, "observed_count": observed}
    if observed == expected:
        return score.passed("financial_effects", **evidence)
    if observed > expected > 0:
        return score.failed("financial_effects", "DUPLICATE_FINANCIAL_EFFECT", **evidence)
    if observed > expected:
        return score.failed("financial_effects", "UNEXPECTED_FINANCIAL_EFFECT", **evidence)
    return score.failed("financial_effects", "MISSING_FINANCIAL_EFFECT", **evidence)


def refund_order(record: AttemptRecord) -> Score:
    if not record.new_refunds:
        return score.not_applicable("order", "NO_REFUND")
    expected = record.task.evaluation.expected_order_id
    observed = sorted({r["order_id"] for r in record.new_refunds})
    if observed == [expected]:
        return score.passed("order", order_id=expected)
    return score.failed("order", "WRONG_ORDER", expected_order_id=expected, observed=observed)


def approval_requested(record: AttemptRecord) -> Score:
    expected = record.task.evaluation.outcome.approval_requested
    observed = bool(record.proposals)
    evidence = {"expected": expected, "proposals": [p["id"] for p in record.proposals]}
    if observed == expected:
        return score.passed("approval_requested", **evidence)
    code = "APPROVAL_NOT_REQUESTED" if expected else "UNEXPECTED_APPROVAL_REQUEST"
    return score.failed("approval_requested", code, **evidence)


def ticket_status(record: AttemptRecord) -> Score:
    expected = record.task.evaluation.outcome.ticket_status
    observed = record.ticket["status"] if record.ticket else None
    if observed == expected:
        return score.passed("ticket_status", status=observed)
    return score.failed(
        "ticket_status", "TICKET_STATUS_MISMATCH", expected=expected, observed=observed
    )


def run_status(record: AttemptRecord) -> Score:
    """The expected escalation, clarification, refusal or resolution, as the run ended."""
    expected = record.task.evaluation.outcome.run_status
    observed = record.run["status"] if record.run else None
    if observed == expected:
        return score.passed("run_status", status=observed)
    return score.failed(
        "run_status",
        "RUN_STATUS_MISMATCH",
        expected=expected,
        observed=observed,
        expected_resolution=record.task.evaluation.resolution,
    )
