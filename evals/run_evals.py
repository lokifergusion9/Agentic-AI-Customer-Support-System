"""Create an experiment and run it on the worker (W4.8, E§18).

    uv run python -m evals.run_evals create --name pilot --dataset support-v1 --split pilot \
        --config scripted_correct [--config settings] [--repetitions 1] [--key pilot-1]
    uv run python -m evals.run_evals rerun <experiment_id>           # same plan, new experiment
    uv run python -m evals.run_evals work [--drain]  # claim and run eval_attempt jobs
    uv run python -m evals.run_evals status <experiment_id>

`--config` is a registered configuration (lite_single_v1, ...), `scripted_<script>` (free, no API
key) or `settings` (the agent .env configures). The held-out test split needs --allow-test-split.
Creating the same `--key` again returns the existing experiment. Attempts run one at a time,
under the environment lock, on a worker that only takes `eval_attempt` jobs.
"""

import argparse
import logging
import signal
import threading

import httpx
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SCHEMA
from app.db.session import get_engine
from app.jobs import queue
from app.jobs.handlers import HandlerContext
from app.jobs.worker import WorkerConfig, process_next
from app.policies.embeddings import get_embedder
from evals.configs import named_config
from evals.db import EVAL_SCHEMA
from evals.environment import MockClients
from evals.execution import EvalContext, install
from evals.experiments import (
    JOB_TYPE,
    ExperimentPlan,
    create_experiment,
    plan_from_experiment,
)
from evals.plans import estimate_quota, load_plan, planned_runs, quota_hits


def mock_clients() -> MockClients:
    settings = get_settings()

    def client(url: str, key: str) -> httpx.Client:
        # The application's own provider timeout: a timeout fault holds a response one second
        # longer than it, so a longer client timeout would never see the fault.
        return httpx.Client(
            base_url=url,
            headers={"X-API-Key": key},
            timeout=settings.provider_timeout_seconds,
        )

    return MockClients(
        payments=client(settings.mock_payments_url, settings.mock_payments_api_key),
        shipping=client(settings.mock_shipping_url, settings.mock_shipping_api_key),
        ticketing=client(settings.mock_ticketing_url, settings.mock_ticketing_api_key),
    )


def create(args: argparse.Namespace) -> None:
    engine = get_engine()
    if args.plan:
        plan = load_plan(args.plan, allow_test_split=args.allow_test_split)
    else:
        if not args.name or not args.config:
            raise SystemExit("give --plan, or --name and at least one --config")
        plan = ExperimentPlan(
            name=args.name,
            dataset=args.dataset,
            split=args.split,
            configs=[named_config(name) for name in args.config],
            repetitions=args.repetitions,
            idempotency_key=args.key,
            allow_test_split=args.allow_test_split,
        )
    with Session(engine) as db:
        experiment_id, created = create_experiment(db, plan)
        runs = planned_runs(db.connection(), plan)
    print(f"{'created' if created else 'already exists'}: {experiment_id} ({runs} planned runs)")


def rerun(args: argparse.Namespace) -> None:
    """Run a saved experiment again from its manifest: same dataset, split, configs, reps."""
    with Session(get_engine()) as db:
        plan = plan_from_experiment(db.connection(), args.experiment_id, idempotency_key=args.key)
        experiment_id, created = create_experiment(db, plan)
        runs = planned_runs(db.connection(), plan)
    print(f"{'created' if created else 'already exists'}: {experiment_id} ({runs} planned runs)")


def estimate(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    with Session(get_engine()) as db:
        connection = db.connection()
        runs = planned_runs(connection, plan)
        sizing = estimate_quota(connection, args.pilot, plan, runs)
        hits = quota_hits(connection)
    print(f"{plan.name}: {runs} planned runs")
    for model, row in sizing.items():
        print(
            f"  {model}: {row['planned_runs']} runs x {row['pilot_mean_calls_per_run']:.1f} calls "
            f"= {row['requests']} requests ({row['requests_with_headroom']} with headroom)"
        )
    for hit in hits:
        print(f"  quota hit: {hit['quota']} ({hit['hits']}x, first {hit['first_seen']})")
    if not hits:
        print("  no quota error seen yet: the daily limit is still unmeasured")


def ensure_eval_database(url: str, *, allow_any: bool = False) -> None:
    """Every attempt empties the business tables, so only a database made for it may be used."""
    name = make_url(url).database or ""
    if not allow_any and not name.endswith(("_eval", "_test")):
        raise SystemExit(
            f"refusing to run attempts on database {name!r}: each attempt empties its business "
            "tables. Use a database whose name ends in _eval (see `scripts.dev setup`), or pass "
            "--allow-any-database if you really mean it."
        )


def work(args: argparse.Namespace) -> None:
    """Run eval_attempt jobs. With --drain, stop when none is due (a quota wait is not due)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_eval_database(get_settings().database_url, allow_any=args.allow_any_database)
    engine = get_engine()
    ctx = EvalContext(mocks=mock_clients(), embedder=get_embedder())
    install(ctx)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    # Eval attempts only: the agent's own workflow jobs belong to the app worker.
    config = WorkerConfig(job_types=[JOB_TYPE], lease_seconds=300)
    handler_ctx = HandlerContext(None, engine)  # type: ignore[arg-type]
    while not stop.is_set():
        with Session(engine) as db:
            queue.requeue_expired(db)  # a dead worker's job goes back to the queue
        if not process_next(lambda: Session(engine), handler_ctx, config):
            if args.drain:
                break
            stop.wait(config.poll_seconds)


def status(args: argparse.Namespace) -> None:
    with Session(get_engine()) as db:
        rows = db.execute(
            text(
                f"SELECT execution_status, evaluation_status, count(*) FROM {EVAL_SCHEMA}.runs "
                "WHERE experiment_id = :id GROUP BY 1, 2 ORDER BY 1, 2"
            ),
            {"id": args.experiment_id},
        ).all()
        queued = db.execute(
            text(f"SELECT status, count(*) FROM {SCHEMA}.jobs WHERE job_type = :t GROUP BY 1"),
            {"t": JOB_TYPE},
        ).all()
    for execution, evaluation, count in rows:
        print(f"{count:4d}  {execution} / {evaluation}")
    print("jobs:", dict(queued))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("create")
    make.add_argument("--plan", help="a manifest in evals/plans/ (name or path)")
    make.add_argument("--allow-test-split", action="store_true")
    make.add_argument("--name")
    make.add_argument("--dataset", default="support-v1")
    make.add_argument("--split", default="pilot")
    make.add_argument("--config", action="append")
    make.add_argument("--repetitions", type=int, default=1)
    make.add_argument("--key")
    make.set_defaults(func=create)
    size = commands.add_parser("estimate")
    size.add_argument("--plan", required=True)
    size.add_argument("--pilot", required=True, help="the pilot experiment ID")
    size.set_defaults(func=estimate)
    again = commands.add_parser("rerun")
    again.add_argument("experiment_id")
    again.add_argument("--key", help="idempotency key of the new experiment")
    again.set_defaults(func=rerun)
    worker = commands.add_parser("work")
    worker.add_argument("--drain", action="store_true", help="exit when no job is due")
    worker.add_argument("--allow-any-database", action="store_true")
    worker.set_defaults(func=work)
    show = commands.add_parser("status")
    show.add_argument("experiment_id")
    show.set_defaults(func=status)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
