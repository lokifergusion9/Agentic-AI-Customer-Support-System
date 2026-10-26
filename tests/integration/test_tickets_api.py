"""POST /tickets takes the customer from the token and checks order ownership (W1.7)."""

from collections.abc import Iterator
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.auth.tokens import mint_token
from app.db.session import get_db
from app.main import app
from scripts.seed_data import seed

MESSAGE = "I returned my headphones last week. When will I get my refund?"


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


def post(client: TestClient, body: dict[str, object], subject: str = "auth|alice"):
    return client.post("/tickets", json=body, headers=bearer(subject))


def count(conn: Connection, table: str) -> int:
    return conn.execute(text(f"SELECT count(*) FROM support.{table}")).scalar_one()


def test_ticket_for_own_order(client: TestClient, conn: Connection) -> None:
    response = post(client, {"message": f"  {MESSAGE}  ", "order_id": "ord_1001"})

    assert response.status_code == 201
    body = response.json()
    assert body["id"].startswith("tkt_")
    assert (body["order_id"], body["status"], body["category"]) == ("ord_1001", "open", None)
    assert "customer_id" not in body

    ticket = conn.execute(
        text("SELECT customer_id, order_id, customer_message FROM support.tickets WHERE id = :id"),
        {"id": body["id"]},
    ).one()
    assert tuple(ticket) == ("cus_alice", "ord_1001", MESSAGE)


def test_ticket_creation_is_audited_without_the_message(
    client: TestClient, conn: Connection
) -> None:
    ticket_id = post(client, {"message": MESSAGE, "order_id": "ord_1001"}).json()["id"]

    event = conn.execute(
        text(
            "SELECT run_id, actor, event_type, redacted_payload FROM support.audit_events "
            "WHERE ticket_id = :id"
        ),
        {"id": ticket_id},
    ).one()
    assert event.run_id is None
    assert (event.actor, event.event_type) == ("customer:cus_alice", "ticket_created")
    assert event.redacted_payload == {"order_id": "ord_1001", "message_length": len(MESSAGE)}
    assert MESSAGE not in str(event.redacted_payload)


def test_ticket_without_an_order(client: TestClient) -> None:
    response = post(client, {"message": "My speaker arrived cracked."}, subject="auth|bob")
    assert response.status_code == 201
    assert response.json()["order_id"] is None


def test_each_ticket_gets_a_new_id(client: TestClient) -> None:
    ids = {post(client, {"message": MESSAGE}).json()["id"] for _ in range(3)}
    assert len(ids) == 3


@pytest.mark.parametrize("order_id", ["ord_1014", "ord_9999"], ids=["other_customer", "missing"])
def test_order_not_found_looks_the_same_and_writes_nothing(
    client: TestClient, conn: Connection, order_id: str
) -> None:
    before = (count(conn, "tickets"), count(conn, "audit_events"))

    response = post(client, {"message": MESSAGE, "order_id": order_id})

    assert response.status_code == 404
    assert response.json() == {
        "success": False,
        "error_code": "ORDER_NOT_FOUND",
        "retryable": False,
        "message": "No order with this ID was found.",
    }
    assert (count(conn, "tickets"), count(conn, "audit_events")) == before


@pytest.mark.parametrize(
    "body",
    [
        {"message": MESSAGE, "customer_id": "cus_bob"},
        {"message": "   "},
        {"message": "x" * 4001},
        {},
        {"message": MESSAGE, "order_id": "1001"},
        {"message": MESSAGE, "order_id": "ord_1001'; DROP TABLE x;--"},
    ],
    ids=["customer_id", "blank", "too_long", "no_message", "bad_prefix", "injection"],
)
def test_invalid_body_is_rejected(client: TestClient, conn: Connection, body: dict) -> None:
    before = count(conn, "tickets")
    assert post(client, body).status_code == 422
    assert count(conn, "tickets") == before


def test_reviewer_cannot_create_tickets(client: TestClient) -> None:
    headers = bearer("auth|reviewer_riya", "reviewer")
    assert client.post("/tickets", json={"message": MESSAGE}, headers=headers).status_code == 403


def test_token_is_required(client: TestClient) -> None:
    assert client.post("/tickets", json={"message": MESSAGE}).status_code == 401


def test_unknown_subject_is_unauthorized(client: TestClient) -> None:
    assert post(client, {"message": MESSAGE}, subject="auth|mallory").status_code == 401
