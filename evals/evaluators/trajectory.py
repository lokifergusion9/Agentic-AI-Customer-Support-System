"""Trajectory evaluator: how the agent got there (W4.9, E§13).

Any valid order passes: required steps are partial-order constraints, not a script. Required
steps are matched one to one, so repeating a correct call never raises recall.
"""

from collections import Counter

from evals.evaluators.base import AttemptRecord, Score, Scorer

NAME, VERSION = "trajectory", "1.1.0"  # 1.1.0: required_steps always reports required+missing
score = Scorer(NAME, VERSION)
# Calls that are repeated by design: reconciliation re-executes and re-checks a refund, and
# the workflow updates the ticket at each pause.
REPEATABLE = {"execute_approved_refund", "get_refund_status", "update_ticket"}


def evaluate(record: AttemptRecord) -> list[Score]:
    return [
        required_steps(record),
        preconditions(record),
        schema_validity(record),
        argument_correctness(record),
        redundant_calls(record),
        irrelevant_calls(record),
        failure_handling(record),
    ]


def required_steps(record: AttemptRecord) -> Score:
    rules = record.task.evaluation.required_behavior
    if not rules:
        return score.not_applicable("required_steps", "NO_REQUIRED_STEPS")
    required = sorted({step for rule in rules for step in (rule.before, *rule.requires)})
    observed = {step for _, step in record.steps()}
    matched = [step for step in required if step in observed]
    missing = [step for step in required if step not in observed]
    recall = len(matched) / len(required)
    evidence = {"required": required, "missing": missing, "recall": recall}
    if not missing:
        return score.passed("required_steps", **evidence)
    if matched:
        return score.partial("required_steps", recall, "MISSING_REQUIRED_STEP", **evidence)
    return score.failed("required_steps", "MISSING_REQUIRED_STEP", **evidence)


def preconditions(record: AttemptRecord) -> Score:
    rules = record.task.evaluation.required_behavior
    if not rules:
        return score.not_applicable("preconditions", "NO_REQUIRED_STEPS")
    steps = record.steps()
    violations = []
    for rule in rules:
        for seq, step in steps:
            if step != rule.before:
                continue
            done_before = {s for q, s in steps if q < seq}
            if unmet := [r for r in rule.requires if r not in done_before]:
                violations.append({"step": step, "sequence_number": seq, "unmet": unmet})
    if violations:
        return score.failed("preconditions", "PRECONDITION_VIOLATED", violations=violations)
    return score.passed("preconditions")


def schema_validity(record: AttemptRecord) -> Score:
    invalid = [
        {"sequence_number": c.sequence_number, "tool": c.tool, "error_code": c.error_code}
        for c in record.tool_calls
        if c.error_code in ("INVALID_ARGUMENTS", "TOOL_NOT_ALLOWED")
    ]
    if invalid:
        return score.failed("schema_validity", "INVALID_TOOL_CALL", calls=invalid)
    return score.passed("schema_validity", calls=len(record.tool_calls))


def argument_correctness(record: AttemptRecord) -> Score:
    """Order IDs are the customer's own or ones they named; a proposal asks the right amount."""
    wrong = []
    for call in record.tool_calls:
        order_id = call.order_id
        if order_id and not record.owns(order_id) and order_id not in record.named_orders:
            wrong.append({"sequence_number": call.sequence_number, "order_id": order_id})
    permitted = record.task.reviewer_behavior
    if permitted is not None:
        for call in record.tool_calls:
            amount = call.args.get("amount_minor")
            if (
                call.tool == "create_refund_proposal"
                and amount != permitted.applies_to.amount_minor
            ):
                wrong.append({"sequence_number": call.sequence_number, "amount_minor": amount})
    if wrong:
        return score.failed("argument_correctness", "WRONG_ARGUMENT", calls=wrong)
    return score.passed("argument_correctness")


def redundant_calls(record: AttemptRecord) -> Score:
    calls = [
        (c.tool, tuple(sorted((k, str(v)) for k, v in c.args.items())))
        for c in record.tool_calls
        if c.outcome == "ok" and c.tool not in REPEATABLE
    ]
    if not calls:
        return score.not_applicable("redundant_calls", "NO_CALLS")
    repeated = {f"{tool}{dict(args)}": n for (tool, args), n in Counter(calls).items() if n > 1}
    if repeated:
        unique = len(set(calls)) / len(calls)
        return score.partial("redundant_calls", unique, "REDUNDANT_CALLS", repeated=repeated)
    return score.passed("redundant_calls")


def irrelevant_calls(record: AttemptRecord) -> Score:
    allowed = set(record.task.evaluation.allowed_actions)
    forbidden = set(record.task.evaluation.forbidden_actions)
    if not allowed:
        return score.not_applicable("irrelevant_calls", "NO_ALLOWED_LIST")
    calls = [c for c in record.tool_calls if c.tool not in forbidden]
    extra = [c.tool for c in calls if c.tool not in allowed]
    if extra:
        share = 1 - len(extra) / len(calls)
        return score.partial(
            "irrelevant_calls", share, "IRRELEVANT_CALLS", tools=sorted(set(extra))
        )
    return score.passed("irrelevant_calls")


def failure_handling(record: AttemptRecord) -> Score:
    """After a tool fails for good, the agent escalates or succeeds on a later try; it never
    carries on as if the call had worked."""
    failed = [c for c in record.tool_calls if c.outcome == "error"]
    retried = len(record.events("tool_retried"))
    if not failed and not record.task.environment_fixture.faults and not retried:
        return score.not_applicable("failure_handling", "NO_FAILURE")
    escalated = record.run is not None and record.run["status"] == "escalated"
    unhandled = [
        c.sequence_number
        for c in failed
        if not escalated
        and not any(
            later.tool == c.tool and later.args == c.args and later.outcome == "ok"
            for later in record.tool_calls
            if later.sequence_number > c.sequence_number
        )
    ]
    if unhandled:
        return score.failed("failure_handling", "FAILURE_NOT_HANDLED", calls=unhandled)
    return score.passed("failure_handling", failed_calls=len(failed), retries=retried)
