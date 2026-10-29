"""Register dated per-model rates so cost can be estimated at paid prices (W5.1, Q4).

    uv run python -m evals.prices add --provider gemini --model gemini-3.5-flash-lite \
        --from 2026-10-01 --input 0.10 --output 0.40 --source "ai.google.dev/pricing, 2026-10-29"

Rates are USD per million tokens and are read from the provider's price page by a person: this
repository does not ship prices. Until a rate is registered for a model, its cost is reported
as unknown (never as free). The free tier itself costs $0; these rates only answer "what would
this have cost at paid rates?" and are always labelled an estimate.
"""

import argparse
from datetime import date

from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.db.session import get_engine
from evals.db import EVAL_SCHEMA


def register_price(
    connection: Connection,
    *,
    provider: str,
    model_id: str,
    effective_from: date,
    input_per_mtok_usd: float,
    output_per_mtok_usd: float,
    source: str,
    cached_input_per_mtok_usd: float | None = None,
) -> None:
    """Add a rate. A rate for the same model and date already registered is refused."""
    if not source.strip():
        raise ValueError("say where the rate was read (URL and date)")
    connection.execute(
        text(
            f"INSERT INTO {EVAL_SCHEMA}.price_versions (provider, model_id, effective_from, "
            "input_per_mtok_usd, output_per_mtok_usd, cached_input_per_mtok_usd, source) "
            "VALUES (:provider, :model, :since, :input, :output, :cached, :source)"
        ),
        {
            "provider": provider,
            "model": model_id,
            "since": effective_from,
            "input": input_per_mtok_usd,
            "output": output_per_mtok_usd,
            "cached": cached_input_per_mtok_usd,
            "source": source,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("--provider", required=True)
    add.add_argument("--model", required=True)
    add.add_argument("--from", dest="since", type=date.fromisoformat, required=True)
    add.add_argument("--input", type=float, required=True, help="USD per million input tokens")
    add.add_argument("--output", type=float, required=True, help="USD per million output tokens")
    add.add_argument("--cached", type=float, default=None)
    add.add_argument("--source", required=True)
    args = parser.parse_args()
    with Session(get_engine()) as db:
        register_price(
            db.connection(),
            provider=args.provider,
            model_id=args.model,
            effective_from=args.since,
            input_per_mtok_usd=args.input,
            output_per_mtok_usd=args.output,
            cached_input_per_mtok_usd=args.cached,
            source=args.source,
        )
        db.commit()
    print(f"registered {args.model} from {args.since}")


if __name__ == "__main__":
    main()
