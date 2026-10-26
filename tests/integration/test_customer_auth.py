"""The authenticated customer comes from the token's `sub`, mapped through the database (W1.5)."""

from collections.abc import Iterator
from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Connection
from sqlalchemy.orm import Session

from app.auth.customer import CurrentCustomer
from app.auth.tokens import mint_token
from app.db.session import get_db
from scripts.seed_data import seed

# A route used only by these tests; real endpoints arrive in W1.7.
app = FastAPI()


@app.get("/whoami")
def whoami(customer: CurrentCustomer) -> dict[str, str]:
    return {"customer_id": customer.customer_id, "subject": customer.subject}


@pytest.fixture
def client(conn: Connection) -> Iterator[TestClient]:
    seed(conn, date(2026, 10, 25))

    def test_db() -> Iterator[Session]:
        with Session(bind=conn) as session:
            yield session

    app.dependency_overrides[get_db] = test_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def bearer(subject: str, role: str = "customer") -> dict[str, str]:
    return {"Authorization": f"Bearer {mint_token(subject, role)}"}


@pytest.mark.parametrize(
    ("subject", "customer_id"),
    [("auth|alice", "cus_alice"), ("auth|bob", "cus_bob"), ("auth|dev", "cus_dev")],
)
def test_seeded_customer_is_resolved(client: TestClient, subject: str, customer_id: str) -> None:
    response = client.get("/whoami", headers=bearer(subject))
    assert response.status_code == 200
    assert response.json() == {"customer_id": customer_id, "subject": subject}


def test_customer_id_in_query_or_body_is_ignored(client: TestClient) -> None:
    response = client.request(
        "GET",
        "/whoami",
        params={"customer_id": "cus_bob"},
        json={"customer_id": "cus_bob"},
        headers=bearer("auth|alice"),
    )
    assert response.json()["customer_id"] == "cus_alice"


def test_unknown_subject_is_unauthorized(client: TestClient) -> None:
    response = client.get("/whoami", headers=bearer("auth|mallory"))
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("role", ["reviewer", "senior_reviewer"])
def test_reviewer_token_is_forbidden(client: TestClient, role: str) -> None:
    assert client.get("/whoami", headers=bearer("auth|reviewer_riya", role)).status_code == 403


def test_missing_header_is_unauthorized(client: TestClient) -> None:
    response = client.get("/whoami")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("header", ["Bearer not-a-jwt", "Basic YWxpY2U6cHc=", "Bearer"])
def test_bad_header_is_unauthorized(client: TestClient, header: str) -> None:
    assert client.get("/whoami", headers={"Authorization": header}).status_code == 401
