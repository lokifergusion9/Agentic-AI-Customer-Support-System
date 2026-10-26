"""Reviewer authentication and authorization (W1.6).

Reviewers are identified by the signed token alone (Q5): `sub` is the reviewer ID that is
saved on approvals, and `role` is `reviewer` or `senior_reviewer`. Reviewer identity and role
come from this dependency, never from the request body or from the model.
"""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status

from app.auth.dependencies import get_token_claims
from app.auth.tokens import TokenClaims

REVIEWER_ROLES = ("reviewer", "senior_reviewer")

# A refund above ₹50,000 needs a senior reviewer (human_review_exceptions_v1:section_3).
# A policy rule, not a deployment setting, so it is not read from the environment.
SENIOR_REVIEW_THRESHOLD_MINOR = 5_000_000


@dataclass(frozen=True)
class AuthenticatedReviewer:
    reviewer_id: str
    role: str


def get_current_reviewer(
    claims: Annotated[TokenClaims, Depends(get_token_claims)],
) -> AuthenticatedReviewer:
    if claims.role not in REVIEWER_ROLES:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Reviewer role required")
    return AuthenticatedReviewer(reviewer_id=claims.subject, role=claims.role)


CurrentReviewer = Annotated[AuthenticatedReviewer, Depends(get_current_reviewer)]


def required_reviewer_role(amount_minor: int) -> str:
    """The role needed to decide a refund of `amount_minor` paise."""
    if amount_minor <= 0:
        raise ValueError(f"refund amount must be positive, got {amount_minor}")
    return "senior_reviewer" if amount_minor > SENIOR_REVIEW_THRESHOLD_MINOR else "reviewer"


def can_decide(reviewer: AuthenticatedReviewer, required_role: str) -> bool:
    """Whether `reviewer` may approve or reject an approval needing `required_role`.

    A senior reviewer may decide anything. A standard reviewer may not decide senior-only
    approvals at all, so those stay queued for a senior reviewer. The database CHECK on
    `approvals` is the backstop for approvals.
    """
    if required_role not in REVIEWER_ROLES:
        raise ValueError(f"unknown required role {required_role!r}")
    return required_role != "senior_reviewer" or reviewer.role == "senior_reviewer"


def ensure_can_decide(reviewer: AuthenticatedReviewer, required_role: str) -> None:
    """Raise 403 unless `reviewer` may decide an approval needing `required_role`."""
    if not can_decide(reviewer, required_role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Senior reviewer approval required"
        )
