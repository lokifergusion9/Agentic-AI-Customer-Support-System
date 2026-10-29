"""Developer commands (W5.8, E§30). The same commands as the Makefile, for Windows without make.

    uv run python -m scripts.dev setup      # locked dependencies, .env, the support_eval database
    uv run python -m scripts.dev migrate    # support schema and eval schema, both databases
    uv run python -m scripts.dev seed       # business data, policies, support-v1, scripted configs
    uv run python -m scripts.dev test       # the deterministic suite (no model calls)
    uv run python -m scripts.dev smoke      # the 5 pilot tasks with the scripted adapter
    uv run python -m scripts.dev evaluate --plan pilot   # a real-model experiment from a manifest
    uv run python -m scripts.dev report <experiment_id>  # export a completed experiment
    uv run python -m scripts.dev demo a|b|c|d|e          # the demo scenarios (scripts/demo.py)

Two databases are used. `DATABASE_URL` in .env is the application database (`support`). Every
experiment runs in its own database named `<name>_eval` (`support_eval`), because each attempt
empties the business tables. `smoke`, `evaluate` and `report` point their child processes at it
themselves, so a forgotten `DATABASE_URL` cannot reach the application database.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url

from app.config import get_settings

ROOT = Path(__file__).resolve().parent.parent
PILOT_SPLIT = "pilot"
SMOKE_CONFIG = "scripted_correct"


def app_url() -> URL:
    return make_url(get_settings().database_url)


def eval_url() -> URL:
    url = app_url()
    name = url.database or "support"
    return url if name.endswith("_eval") else url.set(database=f"{name}_eval")


def render(url: URL) -> str:
    return url.render_as_string(hide_password=False)


def run(command: list[str], database: URL | None = None, check: bool = True) -> int:
    """Run a child process in the project root, optionally pointed at another database."""
    env = dict(os.environ)
    if database is not None:
        env["DATABASE_URL"] = render(database)
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, check=False, cwd=ROOT, env=env)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def python(*args: str, database: URL | None = None, check: bool = True) -> int:
    return run([sys.executable, *args], database, check)


# --- Commands -----------------------------------------------------------------------------------


def setup(_: argparse.Namespace) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is not installed: https://docs.astral.sh/uv/")
    run([uv, "sync", "--locked"])
    env_file = ROOT / ".env"
    if not env_file.exists():
        shutil.copyfile(ROOT / ".env.example", env_file)
        print("created .env from .env.example: add LLM_API_KEY before any real-model run")
    create_database(eval_url())


def create_database(url: URL) -> None:
    """Create the database if it is missing. Needs PostgreSQL up (`docker compose up -d`)."""
    name = url.database
    engine = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": name}
            ).scalar_one_or_none()
            if exists is None:
                connection.execute(text(f'CREATE DATABASE "{name}"'))
                print(f"created database {name}")
            else:
                print(f"database {name} already exists")
    except Exception as error:  # the likely cause is PostgreSQL not running
        raise SystemExit(
            f"cannot reach PostgreSQL ({type(error).__name__}): "
            "start it with `docker compose up -d postgres`"
        ) from error
    finally:
        engine.dispose()


def migrate(_: argparse.Namespace) -> None:
    for url in (app_url(), eval_url()):
        print(f"migrating {url.database}")
        python("-m", "alembic", "upgrade", "head", database=url)


def seed(args: argparse.Namespace) -> None:
    seed_args = [] if args.no_mocks else ["--push-mocks"]
    python("-m", "scripts.seed_data", *seed_args, database=app_url())
    python("-m", "scripts.ingest_policies", database=app_url())
    # The eval database needs the policies the agent retrieves from, and the dataset and the
    # scripted configurations registered. Its business tables are filled per attempt.
    python("-m", "scripts.ingest_policies", database=eval_url())
    python("-m", "scripts.dev", "_register")


def register(_: argparse.Namespace) -> None:
    """Internal: import support-v1 and the scripted configs. Runs against the eval database."""
    from evals.adapters.scripted import SCRIPTS
    from evals.configs import register_agent_config, scripted_config_row
    from evals.experiments import register_dataset
    from evals.loader import load_dataset

    engine = create_engine(eval_url())  # never the application database, whatever the env says
    with engine.begin() as connection:
        tasks = {task.id: task for task in load_dataset("support-v1")}
        dataset = register_dataset(connection, "support-v1", tasks)
        for script in SCRIPTS:
            register_agent_config(connection, scripted_config_row(script))
    engine.dispose()
    print(f"registered support-v1 ({len(tasks)} tasks) and {len(SCRIPTS)} scripted configs")
    print(f"dataset hash {dataset['content_hash']}")


def test(args: argparse.Namespace) -> None:
    python("-m", "pytest", *args.pytest_args)


def smoke(_: argparse.Namespace) -> None:
    url = eval_url()
    key = "smoke-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    python(
        "-m",
        "evals.run_evals",
        "create",
        "--name",
        "smoke",
        "--dataset",
        "support-v1",
        "--split",
        PILOT_SPLIT,
        "--config",
        SMOKE_CONFIG,
        "--key",
        key,
        database=url,
    )
    python("-m", "evals.run_evals", "work", "--drain", database=url)
    engine = create_engine(url)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT r.execution_status, r.evaluation_status, count(*) FROM eval.runs r "
                "JOIN eval.experiments e ON e.id = r.experiment_id "
                "WHERE e.idempotency_key = :key GROUP BY 1, 2 ORDER BY 1, 2"
            ),
            {"key": key},
        ).all()
    engine.dispose()
    for execution, evaluation, count in rows:
        print(f"{count:3d}  {execution} / {evaluation}")
    if [(e, v) for e, v, _ in rows] != [("completed", "pass")]:
        raise SystemExit(
            "smoke failed: every pilot run should be completed / pass. Are the mock services "
            "up (`docker compose --profile mocks up -d`) and the database migrated and seeded?"
        )
    print("smoke passed")


def evaluate(args: argparse.Namespace) -> None:
    url = eval_url()
    create = ["-m", "evals.run_evals", "create", "--plan", args.plan]
    if args.allow_test_split:
        create.append("--allow-test-split")
    python(*create, database=url)
    if not args.create_only:
        python("-m", "evals.run_evals", "work", "--drain", database=url)
        print("runs waiting for a quota resume on the next `evaluate`/`work`; check `status`")


def demo(args: argparse.Namespace) -> None:
    code = python("-m", "scripts.demo", *args.demo_args, database=eval_url(), check=False)
    raise SystemExit(code)


def report(args: argparse.Namespace) -> None:
    command = ["-m", "evals.reports", "export", args.experiment_id]
    if args.out:
        command += ["--out", args.out]
    python(*command, database=eval_url())


# --- Entry point --------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="locked dependencies, .env, the eval database").set_defaults(
        func=setup
    )
    commands.add_parser("migrate", help="apply migrations to both databases").set_defaults(
        func=migrate
    )
    seeding = commands.add_parser("seed", help="business data, policies, dataset, configs")
    seeding.add_argument("--no-mocks", action="store_true", help="do not load the mock services")
    seeding.set_defaults(func=seed)
    commands.add_parser("_register").set_defaults(func=register)
    testing = commands.add_parser("test", help="the deterministic test suite")
    testing.add_argument("pytest_args", nargs="*", help="passed to pytest")
    testing.set_defaults(func=test)
    commands.add_parser("smoke", help="the 5 pilot tasks with the scripted adapter").set_defaults(
        func=smoke
    )
    evaluating = commands.add_parser("evaluate", help="run a real-model experiment plan")
    evaluating.add_argument("--plan", required=True, help="a manifest in evals/plans/")
    evaluating.add_argument("--allow-test-split", action="store_true")
    evaluating.add_argument("--create-only", action="store_true", help="queue it, do not run")
    evaluating.set_defaults(func=evaluate)
    demonstrating = commands.add_parser("demo", help="demo scenarios A-E (see scripts/demo.py)")
    demonstrating.add_argument("demo_args", nargs=argparse.REMAINDER, help="a|b|c|d|e [options]")
    demonstrating.set_defaults(func=demo)
    reporting = commands.add_parser("report", help="export a completed experiment")
    reporting.add_argument("experiment_id")
    reporting.add_argument("--out", default=None)
    reporting.set_defaults(func=report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
