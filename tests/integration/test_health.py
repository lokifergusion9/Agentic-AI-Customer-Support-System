"""GET /health reports the database state without needing a token (W1.7)."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, create_engine
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_healthy_database(client: TestClient, conn: Connection) -> None:
    def test_db() -> Iterator[Session]:
        with Session(bind=conn) as session:
            yield session

    app.dependency_overrides[get_db] = test_db
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"database": "ok"}}


def test_unreachable_database(client: TestClient) -> None:
    # Nothing listens on port 1, so the connection is refused straight away.
    engine = create_engine(
        "postgresql+psycopg://support:support@127.0.0.1:1/support",
        connect_args={"connect_timeout": 1},
    )

    def down_db() -> Iterator[Session]:
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = down_db
    response = client.get("/health")
    engine.dispose()
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "checks": {"database": "unavailable"}}
