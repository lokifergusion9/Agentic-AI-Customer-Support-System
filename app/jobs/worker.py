"""The worker process: claims jobs and runs their handlers (W3.13, Q7, E§17).

    uv run python -m app.jobs.worker

On start it requeues jobs whose worker died and runs the reconciliation sweep (W3.12) before
taking any job; it repeats both every `sweep_seconds`. Each job is claimed in its own short
transaction, a heartbeat thread extends the lease while the handler runs, and the job is
finished only with the current lease token. SIGTERM / Ctrl+C stop it after the current job.
"""

import logging
import os
import signal
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.jobs import queue
from app.jobs.handlers import HANDLERS, HandlerContext
from app.jobs.queue import Job, LeaseLost
from app.workflows.recovery import recover

logger = logging.getLogger(__name__)

SWEEP_LOCK = "worker:sweep"
# Opens a database session: `Session(engine)` in the worker, the test's connection in tests.
Sessions = Callable[[], Session]


@dataclass
class WorkerConfig:
    worker_id: str = field(default_factory=lambda: f"{socket.gethostname()}:{os.getpid()}")
    lease_seconds: float = 60.0
    poll_seconds: float = 1.0
    sweep_seconds: float = 60.0
    job_types: list[str] | None = None


def process_next(sessions: Sessions, ctx: HandlerContext, config: WorkerConfig) -> bool:
    """Claim and run one job; False when there was nothing to do."""
    with sessions() as db:
        job = queue.claim(
            db, config.worker_id, lease_seconds=config.lease_seconds, job_types=config.job_types
        )
    if job is None:
        return False
    logger.info("job %s (%s) attempt %d", job.id, job.job_type, job.attempts)
    stop = threading.Event()
    beat = threading.Thread(target=heartbeat, args=(sessions, job, config, stop), daemon=True)
    beat.start()
    error: str | None = None
    try:
        with sessions() as db:
            HANDLERS[job.job_type](db, job, ctx)
    except Exception as exc:
        logger.exception("job %s failed", job.id)
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        beat.join()
    try:
        with sessions() as db:
            if error is None:
                queue.complete(db, job)
            else:
                queue.fail(db, job, error)
    except LeaseLost:
        # Another worker reclaimed it; that worker finishes it.
        logger.warning("job %s: lease lost, not finalizing", job.id)
    return True


def heartbeat(sessions: Sessions, job: Job, config: WorkerConfig, stop: threading.Event) -> None:
    while not stop.wait(config.lease_seconds / 3):
        try:
            with sessions() as db:
                queue.heartbeat(db, job, lease_seconds=config.lease_seconds)
        except LeaseLost:
            logger.warning("job %s: lease lost during the handler", job.id)
            return


def sweep(sessions: Sessions, ctx: HandlerContext) -> None:
    """Requeue dead workers' jobs, then reconcile and queue unfinished runs (one worker at a time)."""
    with sessions() as db:
        requeued = queue.requeue_expired(db)
    if requeued:
        logger.info("requeued %s", requeued)
    with ctx.engine.connect() as connection:
        key = {"key": SWEEP_LOCK}
        got = connection.execute(
            text("SELECT pg_try_advisory_lock(hashtextextended(:key, 0))"), key
        ).scalar_one()
        connection.commit()
        if not got:
            return  # another worker is sweeping
        try:
            with sessions() as db:
                report = recover(db, ctx.services, queue_work=True)
            logger.info("sweep: %s", report)
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(hashtextextended(:key, 0))"), key)
            connection.commit()


def run_worker(
    sessions: Sessions, ctx: HandlerContext, config: WorkerConfig, stop: threading.Event
) -> None:
    logger.info("worker %s starting", config.worker_id)
    sweep(sessions, ctx)  # reconcile before taking any job (W3.12)
    last_sweep = time.monotonic()
    while not stop.is_set():
        if time.monotonic() - last_sweep >= config.sweep_seconds:
            sweep(sessions, ctx)
            last_sweep = time.monotonic()
        if not process_next(sessions, ctx, config):
            stop.wait(config.poll_seconds)
    logger.info("worker %s stopped", config.worker_id)


def main() -> int:
    from fastapi import HTTPException

    from app.api.runs import get_workflow_services
    from app.db.session import get_engine

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        services = get_workflow_services()
    except HTTPException as exc:
        logger.error("cannot start: %s", exc.detail)
        return 1
    engine = get_engine()
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    ctx = HandlerContext(services, engine)
    run_worker(lambda: Session(engine), ctx, WorkerConfig(), stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
