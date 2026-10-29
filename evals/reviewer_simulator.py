"""Deterministic reviewer simulator (W4.5, E§8).

When a run waits for approval, the simulator decides it the way the task's `reviewer_behavior`
says, through the real approval service, so every check a human reviewer meets applies to it:

- approve: approves only a proposal matching the permitted action (action, order and amount);
  anything else is rejected.
- reject:  rejects.
- expire:  lets a matching proposal's approval expire, by deciding it at a time just past its
  expiry; the service records the expiry, as it would for a reviewer who never came.
- no reviewer_behavior (the task expects no approval): any approval request is rejected.

The simulator's reviewer has the role the approval requires (`sim|reviewer` or
`sim|senior_reviewer`), so the role check is the service's, not ours. Its decisions are
simulated approval outcomes: the `sim|` prefix marks them in the audit trail, and the time
spent deciding is reported as approval wait, apart from the agent's active execution.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.reviewer import AuthenticatedReviewer
from app.db import SCHEMA
from app.services.approvals import decide_approval
from app.services.errors import ServiceFailure
from evals.task_format import ReviewerBehavior

SIMULATOR_REVIEWERS = {
    role: AuthenticatedReviewer(reviewer_id=f"sim|{role}", role=role)
    for role in ("reviewer", "senior_reviewer")
}
# How far past the expiry the simulated clock is moved to let an approval expire.
PAST_EXPIRY = timedelta(seconds=1)


@dataclass(frozen=True)
class SimulatedApproval:
    approval_id: str
    proposal_id: str
    behavior: str  # approve | reject | expire | none (the task expects no approval)
    matched: bool  # the proposal is the permitted action
    outcome: str  # approved | rejected | expired
    required_role: str
    reviewer_id: str
    wait_ms: float  # real time spent deciding
    simulated_delay_seconds: float  # how far the clock was moved; 0 unless expired

    def as_event(self) -> dict[str, Any]:
        return {"simulated": True, **self.__dict__}


@dataclass
class ReviewerSimulator:
    behavior: ReviewerBehavior | None
    clock: Callable[[], float] = time.monotonic
    decisions: list[SimulatedApproval] = field(default_factory=list)

    @property
    def total_wait_ms(self) -> float:
        return sum(d.wait_ms for d in self.decisions)

    def pending_approval(self, db: Session, run_id: str) -> str | None:
        """The run's undecided approval, if it is waiting for one."""
        return db.execute(
            text(
                f"SELECT a.id FROM {SCHEMA}.approvals a "
                f"JOIN {SCHEMA}.action_proposals p ON p.id = a.proposal_id "
                "WHERE p.run_id = :run AND a.decision IS NULL "
                "ORDER BY a.created_at DESC, a.id DESC LIMIT 1"
            ),
            {"run": run_id},
        ).scalar_one_or_none()

    def decide(
        self, db: Session, approval_id: str, *, now: datetime | None = None
    ) -> SimulatedApproval:
        started = self.clock()
        now = now or datetime.now(UTC)
        approval = db.execute(
            text(
                "SELECT a.id, a.proposal_id, a.required_role, a.expires_at, p.action_type, "
                "p.order_id, p.payload "
                f"FROM {SCHEMA}.approvals a "
                f"JOIN {SCHEMA}.action_proposals p ON p.id = a.proposal_id "
                "WHERE a.id = :id"
            ),
            {"id": approval_id},
        ).one()
        matched = self.matches(approval)
        behavior = self.behavior.decision if self.behavior else "none"
        reviewer = SIMULATOR_REVIEWERS[approval.required_role]

        delay = 0.0
        if behavior == "expire" and matched:
            later = approval.expires_at + PAST_EXPIRY
            delay = max((later - now).total_seconds(), 0.0)
            try:
                decide_approval(db, reviewer, approval_id, "approved", now=later)
            except ServiceFailure as failure:
                if failure.error_code != "APPROVAL_EXPIRED":
                    raise
            else:
                raise AssertionError("an approval decided past its expiry was not expired")
            outcome = "expired"
        else:
            outcome = "approved" if behavior == "approve" and matched else "rejected"
            decide_approval(db, reviewer, approval_id, outcome, now=now)

        decision = SimulatedApproval(
            approval_id=approval_id,
            proposal_id=approval.proposal_id,
            behavior=behavior,
            matched=matched,
            outcome=outcome,
            required_role=approval.required_role,
            reviewer_id=reviewer.reviewer_id,
            wait_ms=(self.clock() - started) * 1000,
            simulated_delay_seconds=delay,
        )
        self.decisions.append(decision)
        return decision

    def matches(self, approval: Any) -> bool:
        if self.behavior is None:
            return False
        permitted = self.behavior.applies_to
        return (approval.action_type, approval.order_id, approval.payload["amount_minor"]) == (
            permitted.action,
            permitted.order_id,
            permitted.amount_minor,
        )
