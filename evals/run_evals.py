"""Create an experiment and run it on the worker (W4.8, E§18).

    uv run python -m evals.run_evals create --name pilot --dataset support-v1 --split pilot \
        --config scripted_correct [--config settings] [--repetitions 1] [--key pilot-1]
    uv run python -m evals.run_evals work            # claim and run eval_attempt jobs
    uv run python -m evals.run_evals status <experiment_id>

`--config` is `scripted_<script>` (free, no API key) or `settings` (the agent .env configures).
Creating the same `--key` again returns the existing experiment. Attempts run one at a time,
under the environment lock, on a worker that only takes `eval_attempt` jobs.
"""

import argparse
import logging
import signal
import threading

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SCHEMA
from app.db.session import get_engine
from app.jobs.handlers import HandlerContext
from app.jobs.worker import WorkerConfig, run_worker
from app.policies.embeddings import get_embedder
from evals.configs import named_config
from evals.db import EVAL_SCHEMA
from evals.environment import MockClients
from evals.execution import EvalContext, install
from evals.experiments import JOB_TYPE, ExperimentPlan, create_experiment


def mock_clients() -> MockClients:
    settings = get_settings()

    def client(url: str, key: str) -> httpx.Client:
        return httpx.Client(base_url=url, headers={"X-API-Key": key}, timeout=10)

    return MockClients(
        payments=client(settings.mock_payments_url, settings.mock_payments_api_key),
        shipping=client(settings.mock_shipping_url, settings.mock_shipping_api_key),
        ticketing=client(settings.mock_ticketing_url, settings.mock_ticketing_api_key),
    )


def create(args: argparse.Namespace) -> None:
    engine = get_engine()
    plan = ExperimentPlan(
        name=args.name,
        dataset=args.dataset,
        split=args.split,
        configs=[named_config(name) for name in args.config],
        repetitions=args.repetitions,
        idempotency_key=args.key,
    )
    with Session(engine) as db:
        experiment_id, created = create_experiment(db, plan)
    print(f"{'created' if created else 'already exists'}: {experiment_id}")


def work(_: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = get_engine()
    ctx = EvalContext(mocks=mock_clients(), embedder=get_embedder())
    install(ctx)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    # Eval attempts only: the agent's own workflow jobs belong to the app worker.
    config = WorkerConfig(job_types=[JOB_TYPE], lease_seconds=300)
    run_worker(lambda: Session(engine), HandlerContext(None, engine), config, stop)  # type: ignore[arg-type]


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
    make.add_argument("--name", required=True)
    make.add_argument("--dataset", default="support-v1")
    make.add_argument("--split", default="pilot")
    make.add_argument("--config", action="append", required=True)
    make.add_argument("--repetitions", type=int, default=1)
    make.add_argument("--key")
    make.set_defaults(func=create)
    commands.add_parser("work").set_defaults(func=work)
    show = commands.add_parser("status")
    show.add_argument("experiment_id")
    show.set_defaults(func=status)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
