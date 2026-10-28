"""The PostgreSQL job queue: claim, lease, finalize, sweep (W3.13)."""

import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.db import SCHEMA
from app.jobs import queue
from app.jobs.queue import LeaseLost

LEASE = 30.0
LATER = datetime.now(UTC) + timedelta(hours=1)


@pytest.fixture
def db(conn: Connection) -> Iterator[Session]:
    conn.execute(text(f"DELETE FROM {SCHEMA}.jobs"))
    with Session(bind=conn, join_transaction_mode="create_savepoint") as session:
        yield session


def status(db: Session, job_id: str) -> tuple[str, int]:
    row = db.execute(
        text(f"SELECT status, attempts FROM {SCHEMA}.jobs WHERE id = :id"), {"id": job_id}
    ).one()
    return row.status, row.attempts


def test_a_job_is_claimed_once_and_completed_with_its_lease(db: Session) -> None:
    job_id = queue.enqueue(db, "workflow_run", {"run_id": "run_1"})
    db.commit()

    job = queue.claim(db, "worker-a", lease_seconds=LEASE)

    assert job is not None and (job.id, job.payload, job.attempts) == (
        job_id,
        {"run_id": "run_1"},
        1,
    )
    assert queue.claim(db, "worker-b", lease_seconds=LEASE) is None  # nothing else to take
    queue.complete(db, job)
    assert status(db, job_id) == ("completed", 1)


def test_a_stale_worker_cannot_finish_a_reclaimed_job(db: Session) -> None:
    queue.enqueue(db, "workflow_run", {})
    db.commit()
    stale = queue.claim(db, "worker-a", lease_seconds=LEASE)
    assert stale is not None

    assert queue.requeue_expired(db, now=LATER) == [stale.id]  # worker-a stopped heartbeating
    fresh = queue.claim(db, "worker-b", lease_seconds=LEASE)
    assert fresh is not None and fresh.id == stale.id and fresh.lease_token != stale.lease_token

    for finalize in (queue.complete, lambda d, j: queue.fail(d, j, "boom")):
        with pytest.raises(LeaseLost):
            finalize(db, stale)
    with pytest.raises(LeaseLost):
        queue.heartbeat(db, stale, lease_seconds=LEASE)
    queue.heartbeat(db, fresh, lease_seconds=LEASE)
    queue.complete(db, fresh)
    assert status(db, fresh.id) == ("completed", 2)


def test_a_failed_attempt_is_retried_later_then_failed(db: Session) -> None:
    job_id = queue.enqueue(db, "workflow_run", {})
    db.commit()
    for attempt in range(1, 4):
        db.execute(
            text(f"UPDATE {SCHEMA}.jobs SET available_at = now() WHERE id = :id"), {"id": job_id}
        )
        job = queue.claim(db, "worker-a", lease_seconds=LEASE)
        assert job is not None and job.attempts == attempt
        queue.fail(db, job, f"error {attempt}")

    assert status(db, job_id) == ("failed", 3)


def test_a_retried_job_waits_for_its_backoff(db: Session) -> None:
    queue.enqueue(db, "workflow_run", {})
    db.commit()
    job = queue.claim(db, "worker-a", lease_seconds=LEASE)
    assert job is not None
    queue.fail(db, job, "temporary")

    assert queue.claim(db, "worker-a", lease_seconds=LEASE) is None  # not available yet


def test_a_job_that_keeps_losing_its_worker_is_failed(db: Session) -> None:
    job_id = queue.enqueue(db, "workflow_run", {})
    db.commit()
    for _ in range(3):
        assert queue.claim(db, "worker-a", lease_seconds=LEASE) is not None
        queue.requeue_expired(db, now=LATER)

    assert status(db, job_id) == ("failed", 3)


def test_one_active_job_per_dedupe_key(db: Session) -> None:
    first = queue.enqueue(db, "workflow_run", {"n": 1}, dedupe_key="run:run_1")
    assert queue.enqueue(db, "workflow_run", {"n": 2}, dedupe_key="run:run_1") is None
    assert queue.active_job(db, "run:run_1") == first
    db.commit()
    job = queue.claim(db, "worker-a", lease_seconds=LEASE)
    assert job is not None
    queue.complete(db, job)

    # Finished jobs no longer block the key.
    assert queue.enqueue(db, "workflow_run", {"n": 3}, dedupe_key="run:run_1") is not None


def test_claim_can_be_limited_to_job_types(db: Session) -> None:
    queue.enqueue(db, "eval_attempt", {})
    db.commit()
    assert queue.claim(db, "worker-a", lease_seconds=LEASE, job_types=["workflow_run"]) is None
    assert queue.claim(db, "worker-a", lease_seconds=LEASE, job_types=["eval_attempt"])


def test_unknown_job_types_are_rejected(db: Session) -> None:
    with pytest.raises(DBAPIError, match="ck_jobs_job_type"), db.begin_nested():
        queue.enqueue(db, "send_email", {})


def test_concurrent_workers_never_claim_the_same_job(engine: Engine) -> None:
    """Real separate connections: SKIP LOCKED hands each job to exactly one worker."""
    with engine.begin() as setup:
        setup.execute(text(f"DELETE FROM {SCHEMA}.jobs"))
        for n in range(20):
            setup.execute(
                text(
                    f"INSERT INTO {SCHEMA}.jobs (id, job_type, payload) VALUES (:id, 'workflow_run', '{{}}')"
                ),
                {"id": f"job_{n:02d}"},
            )
    claimed: list[str] = []
    lock = threading.Lock()
    start = threading.Barrier(4)

    def worker(name: str) -> None:
        with Session(engine) as db:
            start.wait()
            while (job := queue.claim(db, name, lease_seconds=LEASE)) is not None:
                with lock:
                    claimed.append(job.id)
                queue.complete(db, job)

    try:
        threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert sorted(claimed) == [f"job_{n:02d}" for n in range(20)]
    finally:
        with engine.begin() as cleanup:
            cleanup.execute(text(f"DELETE FROM {SCHEMA}.jobs"))
