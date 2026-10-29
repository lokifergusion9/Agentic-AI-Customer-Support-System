"""The PostgreSQL job queue (W3.13, Q7, E§17).

- `enqueue` adds a job in the caller's transaction, so a run and its job commit together.
- `claim` takes the oldest available job with `FOR UPDATE SKIP LOCKED` in its own short
  transaction and commits before the handler makes any model or provider call.
- The claiming worker holds a lease. `heartbeat` extends it; `complete` and `fail` only
  take effect with the current lease token, so a stale worker cannot finish a reclaimed job.
- `requeue_expired` puts jobs whose lease ran out back in the queue (or fails them after
  their last attempt).
"""

import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import SCHEMA

# Wait before retrying a failed attempt: 5 s, 25 s, ...
RETRY_BASE_SECONDS = 5


@dataclass(frozen=True)
class Job:
    id: str
    job_type: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int
    lease_token: str


class LeaseLost(Exception):
    """The job was reclaimed by another worker; this worker must not finish it."""


class Deferred(Exception):
    """Raised by a handler: nothing is wrong with the job, it cannot run before `available_at`.

    The worker puts the job back in the queue without counting the attempt, so a long wait
    (an exhausted daily quota, say) never uses up the job's retries.
    """

    def __init__(self, available_at: datetime, reason: str) -> None:
        super().__init__(reason)
        self.available_at = available_at
        self.reason = reason


def enqueue(
    db: Session,
    job_type: str,
    payload: dict[str, Any],
    *,
    dedupe_key: str | None = None,
    available_at: datetime | None = None,
) -> str | None:
    """Add a job in the caller's transaction; returns its ID.

    With a `dedupe_key`, nothing is added while a job with the same key is queued or running,
    and None is returned: that job already covers the work.
    """
    job_id = f"job_{secrets.token_hex(8)}"
    inserted = db.execute(
        text(
            f"INSERT INTO {SCHEMA}.jobs (id, job_type, payload, dedupe_key, available_at) "
            "VALUES (:id, :type, CAST(:payload AS jsonb), :dedupe, COALESCE(:at, now())) "
            "ON CONFLICT (dedupe_key) WHERE status IN ('queued', 'running') DO NOTHING "
            "RETURNING id"
        ),
        {
            "id": job_id,
            "type": job_type,
            "payload": json.dumps(payload),
            "dedupe": dedupe_key,
            "at": available_at,
        },
    ).scalar_one_or_none()
    return inserted


def claim(
    db: Session, worker_id: str, *, lease_seconds: float, job_types: list[str] | None = None
) -> Job | None:
    """Take the oldest available job and lease it to `worker_id`; commits immediately."""
    token = secrets.token_hex(16)
    row = db.execute(
        text(
            f"UPDATE {SCHEMA}.jobs SET status = 'running', worker_id = :worker, "
            "lease_token = :token, lease_expires_at = now() + make_interval(secs => :lease), "
            "attempts = attempts + 1 "
            f"WHERE id = (SELECT id FROM {SCHEMA}.jobs WHERE status = 'queued' "
            "AND available_at <= now() "
            "AND (CAST(:types AS text[]) IS NULL OR job_type = ANY(CAST(:types AS text[]))) "
            "ORDER BY available_at, created_at LIMIT 1 FOR UPDATE SKIP LOCKED) "
            "RETURNING id, job_type, payload, attempts, max_attempts, lease_token"
        ),
        {"worker": worker_id, "token": token, "lease": lease_seconds, "types": job_types},
    ).one_or_none()
    db.commit()
    return None if row is None else Job(**row._mapping)


def heartbeat(db: Session, job: Job, *, lease_seconds: float) -> None:
    """Extend the lease; raises LeaseLost if another worker has the job now."""
    extended = db.execute(
        text(
            f"UPDATE {SCHEMA}.jobs SET lease_expires_at = now() + make_interval(secs => :lease) "
            "WHERE id = :id AND status = 'running' AND lease_token = :token RETURNING id"
        ),
        {"id": job.id, "token": job.lease_token, "lease": lease_seconds},
    ).scalar_one_or_none()
    db.commit()
    if extended is None:
        raise LeaseLost(job.id)


def complete(db: Session, job: Job) -> None:
    finish(db, job, "completed", error=None)


def fail(db: Session, job: Job, error: str) -> None:
    """Retry later if attempts are left, otherwise mark the job failed."""
    if job.attempts < job.max_attempts:
        delay = timedelta(seconds=RETRY_BASE_SECONDS * 5 ** (job.attempts - 1))
        retried = db.execute(
            text(
                f"UPDATE {SCHEMA}.jobs SET status = 'queued', worker_id = NULL, "
                "lease_token = NULL, lease_expires_at = NULL, last_error = :error, "
                "available_at = now() + make_interval(secs => :delay) "
                "WHERE id = :id AND status = 'running' AND lease_token = :token RETURNING id"
            ),
            {
                "id": job.id,
                "token": job.lease_token,
                "error": error[:2000],
                "delay": delay.total_seconds(),
            },
        ).scalar_one_or_none()
        db.commit()
        if retried is None:
            raise LeaseLost(job.id)
        return
    finish(db, job, "failed", error=error)


def defer(db: Session, job: Job, *, available_at: datetime, reason: str) -> None:
    """Put a running job back in the queue for later, giving the attempt back."""
    deferred = db.execute(
        text(
            f"UPDATE {SCHEMA}.jobs SET status = 'queued', worker_id = NULL, lease_token = NULL, "
            "lease_expires_at = NULL, attempts = GREATEST(attempts - 1, 0), "
            "available_at = :at, last_error = :reason "
            "WHERE id = :id AND status = 'running' AND lease_token = :token RETURNING id"
        ),
        {"id": job.id, "token": job.lease_token, "at": available_at, "reason": reason[:2000]},
    ).scalar_one_or_none()
    db.commit()
    if deferred is None:
        raise LeaseLost(job.id)


def finish(db: Session, job: Job, status: str, *, error: str | None) -> None:
    finished = db.execute(
        text(
            f"UPDATE {SCHEMA}.jobs SET status = :status, worker_id = NULL, lease_token = NULL, "
            "lease_expires_at = NULL, finished_at = now(), "
            "last_error = COALESCE(:error, last_error) "
            "WHERE id = :id AND status = 'running' AND lease_token = :token RETURNING id"
        ),
        {"id": job.id, "token": job.lease_token, "status": status, "error": error and error[:2000]},
    ).scalar_one_or_none()
    db.commit()
    if finished is None:
        raise LeaseLost(job.id)


def requeue_expired(db: Session, *, now: datetime | None = None) -> list[str]:
    """Jobs whose worker stopped heartbeating go back to the queue; returns their IDs.

    A job already on its last attempt is failed instead, so a job that keeps killing its
    worker cannot loop forever.
    """
    params = {"now": now or datetime.now(UTC)}
    db.execute(
        text(
            f"UPDATE {SCHEMA}.jobs SET status = 'failed', worker_id = NULL, lease_token = NULL, "
            "lease_expires_at = NULL, finished_at = now(), "
            "last_error = 'lease expired on the last attempt' "
            "WHERE status = 'running' AND lease_expires_at < :now AND attempts >= max_attempts"
        ),
        params,
    )
    requeued = db.execute(
        text(
            f"UPDATE {SCHEMA}.jobs SET status = 'queued', worker_id = NULL, lease_token = NULL, "
            "lease_expires_at = NULL, available_at = now(), last_error = 'lease expired' "
            "WHERE status = 'running' AND lease_expires_at < :now RETURNING id"
        ),
        params,
    ).scalars()
    ids = list(requeued)
    db.commit()
    return ids


def active_job(db: Session, dedupe_key: str) -> str | None:
    """The queued or running job for this key, if any."""
    return db.execute(
        text(
            f"SELECT id FROM {SCHEMA}.jobs WHERE dedupe_key = :key "
            "AND status IN ('queued', 'running')"
        ),
        {"key": dedupe_key},
    ).scalar_one_or_none()
