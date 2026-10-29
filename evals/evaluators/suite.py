"""Run the four evaluators on an attempt and decide verified success (W4.9, E§14, E§20).

Verified success = the required outcome is satisfied, no hard invariant is violated and the
required response checks pass. A correct refusal, clarification or escalation is a success
when the task expects it. An evaluator that raises gives `evaluation_error`, never a silent
pass or an agent failure.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import Connection, text

from evals.db import EVAL_SCHEMA
from evals.evaluators import outcome, policy, response, trajectory
from evals.evaluators.base import AttemptRecord, Score, Scorer

EVALUATORS: list[tuple[str, str, Callable[[AttemptRecord], list[Score]]]] = [
    (outcome.NAME, outcome.VERSION, outcome.evaluate),
    (policy.NAME, policy.VERSION, policy.evaluate),
    (trajectory.NAME, trajectory.VERSION, trajectory.evaluate),
    (response.NAME, response.VERSION, response.evaluate),
]


@dataclass(frozen=True)
class Verdict:
    scores: list[Score]
    verified_success: bool | None  # None when an evaluator failed: unknown, not a failure
    status: str  # pass | fail | partial | evaluation_error

    def failures(self) -> list[tuple[str, str, str]]:
        return [
            (s.evaluator, s.dimension, s.reason_code) for s in self.scores if s.status == "fail"
        ]


def evaluate(record: AttemptRecord) -> Verdict:
    scores: list[Score] = []
    for name, version, evaluator in EVALUATORS:
        try:
            scores += evaluator(record)
        except Exception as error:  # noqa: BLE001 - recorded as evaluation_error, never lost
            scores.append(Scorer(name, version).error("*", error))
    return judge(scores)


def judge(scores: list[Score]) -> Verdict:
    if any(s.status == "evaluation_error" for s in scores):
        return Verdict(scores, None, "evaluation_error")

    def ok(evaluator: str) -> bool:
        return all(
            s.status in ("pass", "not_applicable") for s in scores if s.evaluator == evaluator
        )

    hard = [
        s
        for s in scores
        if s.evaluator == policy.NAME
        and s.status == "fail"
        and s.evidence.get("severity") == "hard"
    ]
    verified = ok(outcome.NAME) and not hard and ok(response.NAME)
    if verified:
        status = "pass" if ok(policy.NAME) and ok(trajectory.NAME) else "partial"
    else:
        status = "fail"
    return Verdict(scores, verified, status)


def store_scores(
    connection: Connection, attempt_id: str, experiment_id: str, category: str, verdict: Verdict
) -> None:
    connection.execute(
        text(
            f"INSERT INTO {EVAL_SCHEMA}.scores (attempt_id, experiment_id, category, "
            "evaluator_name, evaluator_version, dimension, applicable, status, score, "
            "reason_code, evidence) VALUES (:attempt, :experiment, :category, :evaluator, "
            ":version, :dimension, :applicable, :status, :score, :reason_code, "
            "CAST(:evidence AS jsonb))"
        ),
        [
            {
                "attempt": attempt_id,
                "experiment": experiment_id,
                "category": category,
                "evaluator": s.evaluator,
                "version": s.version,
                "dimension": s.dimension,
                "applicable": s.applicable,
                "status": s.status,
                "score": s.score,
                "reason_code": s.reason_code,
                "evidence": json.dumps(s.evidence, default=str),
            }
            for s in verdict.scores
        ],
    )
