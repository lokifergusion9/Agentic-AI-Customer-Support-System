"""Reviewer identity comes from the token, and high-value refunds need a senior reviewer (W1.6)."""

from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.auth.reviewer import (
    REVIEWER_ROLES,
    SENIOR_REVIEW_THRESHOLD_MINOR,
    AuthenticatedReviewer,
    CurrentReviewer,
    ensure_can_decide,
    required_reviewer_role,
)
from app.auth.tokens import mint_token

POLICY = Path(__file__).resolve().parents[2] / "app/policies/documents"

# Routes used only by these tests; the approval endpoints arrive in W3.6.
app = FastAPI()


@app.get("/whoami")
def whoami(reviewer: CurrentReviewer) -> dict[str, str]:
    return {"reviewer_id": reviewer.reviewer_id, "role": reviewer.role}


@app.post("/decide")
def decide(reviewer: CurrentReviewer, amount_minor: int) -> dict[str, str]:
    ensure_can_decide(reviewer, required_reviewer_role(amount_minor))
    return {"decided_by": reviewer.reviewer_id}


client = TestClient(app)


def bearer(subject: str, role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {mint_token(subject, role)}"}


@pytest.mark.parametrize("role", REVIEWER_ROLES)
def test_reviewer_identity_comes_from_the_token(role: str) -> None:
    response = client.get(
        "/whoami",
        params={"reviewer_id": "auth|someone_else", "role": "senior_reviewer"},
        headers=bearer("auth|reviewer_riya", role),
    )
    assert response.status_code == 200
    assert response.json() == {"reviewer_id": "auth|reviewer_riya", "role": role}


def test_customer_token_is_forbidden() -> None:
    assert client.get("/whoami", headers=bearer("auth|alice", "customer")).status_code == 403


@pytest.mark.parametrize("header", [None, "Bearer not-a-jwt"])
def test_missing_or_bad_token_is_unauthorized(header: str | None) -> None:
    headers = {"Authorization": header} if header else {}
    response = client.get("/whoami", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    ("amount_minor", "role"),
    [
        (1, "reviewer"),
        (249_900, "reviewer"),
        (SENIOR_REVIEW_THRESHOLD_MINOR, "reviewer"),  # exactly ₹50,000 is not "more than"
        (SENIOR_REVIEW_THRESHOLD_MINOR + 1, "senior_reviewer"),
        (8_999_900, "senior_reviewer"),
    ],
)
def test_required_role_by_amount(amount_minor: int, role: str) -> None:
    assert required_reviewer_role(amount_minor) == role


@pytest.mark.parametrize("amount_minor", [0, -100])
def test_required_role_rejects_non_positive_amounts(amount_minor: int) -> None:
    with pytest.raises(ValueError):
        required_reviewer_role(amount_minor)


def test_threshold_matches_the_policy_document() -> None:
    text = (POLICY / "human_review_exceptions_v1.md").read_text(encoding="utf-8")
    assert f"more than ₹{SENIOR_REVIEW_THRESHOLD_MINOR // 100:,}" in text


@pytest.mark.parametrize(
    ("reviewer_role", "required_role", "allowed"),
    [
        ("reviewer", "reviewer", True),
        ("reviewer", "senior_reviewer", False),
        ("senior_reviewer", "reviewer", True),
        ("senior_reviewer", "senior_reviewer", True),
    ],
)
def test_permission_matrix(reviewer_role: str, required_role: str, allowed: bool) -> None:
    reviewer = AuthenticatedReviewer("auth|reviewer_riya", reviewer_role)
    if allowed:
        ensure_can_decide(reviewer, required_role)
    else:
        with pytest.raises(HTTPException) as error:
            ensure_can_decide(reviewer, required_role)
        assert error.value.status_code == 403


def test_unknown_required_role_is_an_error() -> None:
    with pytest.raises(ValueError):
        ensure_can_decide(AuthenticatedReviewer("auth|reviewer_riya", "senior_reviewer"), "admin")


@pytest.mark.parametrize(
    ("role", "amount_minor", "status"),
    [
        ("reviewer", 249_900, 200),
        ("reviewer", 8_999_900, 403),
        ("senior_reviewer", 8_999_900, 200),
        ("customer", 249_900, 403),
    ],
)
def test_decision_route(role: str, amount_minor: int, status: int) -> None:
    response = client.post(
        "/decide", params={"amount_minor": amount_minor}, headers=bearer("auth|riya", role)
    )
    assert response.status_code == status
