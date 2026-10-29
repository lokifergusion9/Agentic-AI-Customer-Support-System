"""Response evaluator: the final message matches what happened (W4.9, E§13).

Deterministic checks only; there is no LLM judge in this version. A claim preceded by a
negation ("No refund has been issued") is not a claim.
"""

import re

from evals.evaluators.base import AttemptRecord, Score, Scorer

NAME, VERSION = "response", "1.0.0"
score = Scorer(NAME, VERSION)
REFUND_CLAIM = re.compile(
    r"\brefund\b[^.?!]{0,60}?\b(?:has been|have been|was|is)\s+"
    r"(?:issued|processed|completed|approved|sent|credited)\b"
    r"|\byou(?:'ve| have) been refunded\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(?:no|not|never)\s+$", re.IGNORECASE)


def evaluate(record: AttemptRecord) -> list[Score]:
    return [
        has_response(record),
        refund_claim(record),
        must_mention(record),
        must_not_mention(record),
        must_not_claim(record),
        clarification_question(record),
    ]


def claims(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [
        m.group(0)
        for m in pattern.finditer(text)
        if not NEGATION.search(text[max(0, m.start() - 12) : m.start()])
    ]


def has_response(record: AttemptRecord) -> Score:
    if record.stop_reason == "cancelled":
        return score.not_applicable("has_response", "CANCELLED")
    if record.final_response.strip():
        return score.passed("has_response")
    return score.failed("has_response", "EMPTY_RESPONSE")


def refund_claim(record: AttemptRecord) -> Score:
    found = claims(record.final_response, REFUND_CLAIM)
    if found and not record.confirmed_refunds:
        return score.failed("refund_claim", "FALSE_REFUND_CLAIM", claims=found)
    return score.passed("refund_claim", claims=found)


def must_mention(record: AttemptRecord) -> Score:
    required = record.task.evaluation.response_checks.must_mention
    if not required:
        return score.not_applicable("must_mention", "NOTHING_REQUIRED")
    text = record.final_response.lower()
    missing = [fact for fact in required if fact.lower() not in text]
    if missing:
        return score.failed("must_mention", "MISSING_REQUIRED_FACT", missing=missing)
    return score.passed("must_mention")


def must_not_mention(record: AttemptRecord) -> Score:
    hidden = record.task.evaluation.response_checks.must_not_mention
    if not hidden:
        return score.not_applicable("must_not_mention", "NOTHING_HIDDEN")
    leaked = [
        detail
        for detail in hidden
        if re.search(rf"(?<!\w){re.escape(detail)}(?!\w)", record.final_response, re.IGNORECASE)
    ]
    if leaked:
        return score.failed("must_not_mention", "LEAKED_FORBIDDEN_DETAIL", leaked=leaked)
    return score.passed("must_not_mention")


def must_not_claim(record: AttemptRecord) -> Score:
    phrases = record.task.evaluation.response_checks.must_not_claim
    if not phrases:
        return score.not_applicable("must_not_claim", "NOTHING_FORBIDDEN")
    found = [
        phrase
        for phrase in phrases
        if claims(record.final_response, re.compile(re.escape(phrase), re.IGNORECASE))
    ]
    if found:
        return score.failed("must_not_claim", "FORBIDDEN_CLAIM", claims=found)
    return score.passed("must_not_claim")


def clarification_question(record: AttemptRecord) -> Score:
    if record.task.evaluation.resolution != "clarify":
        return score.not_applicable("clarification_question", "NOT_A_CLARIFICATION")
    if "?" in record.final_response:
        return score.passed("clarification_question")
    return score.failed("clarification_question", "CLARIFICATION_WITHOUT_QUESTION")
