"""The five demo scenarios (W5.7, §18, E§31), each checked against the system state.

    uv run python -m scripts.dev demo a       # a successful approved refund
    uv run python -m scripts.dev demo b       # a correctly blocked request
    uv run python -m scripts.dev demo c       # failure recovery: accepted refund, lost response
    uv run python -m scripts.dev demo d       # a convincing but incorrect answer
    uv run python -m scripts.dev demo e --baseline EXP --candidate EXP   # prompt v1 vs v2

A, B and C run the `scripted_correct` agent by default: no API key, deterministic. It works
only from what the tools return, but it does not retrieve policy, so the policy step shows as
"not done by the scripted agent". Pass `--config lite_specialist_v1` to run the real model
(needs `LLM_API_KEY` and free-tier quota). D uses the scripted false-success agent by default
and is labeled as scripted. E needs two finished real-model experiments (W5.2) and does not
fake them.

Every demo empties the business tables of the database it runs on, so it only runs on a
database named `*_eval` or `*_test` (`scripts.dev demo` points it at `support_eval`).
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SCHEMA
from app.db.session import get_engine
from app.policies.embeddings import get_embedder
from app.services.providers import ProviderClient
from app.services.refunds import confirm_refund
from evals.adapters.base import AttemptContext
from evals.comparison import IncompatibleExperiments, Side, compare
from evals.configs import named_config
from evals.environment import environment_lock, prepare, run_trace, snapshot
from evals.evaluators.base import AttemptRecord
from evals.evaluators.suite import Verdict, evaluate
from evals.execution import default_adapter, lock_connection
from evals.loader import load_dataset
from evals.reviewer_simulator import ReviewerSimulator
from evals.run_evals import ensure_eval_database, mock_clients

DATASET = "support-v1"
TASKS = {
    "a": "refund_eligible_001",
    "b": "ineligible_return_in_transit_001",
    "c": "fault_refund_accept_timeout_001",
    "d": "refund_eligible_001",
}
DEFAULT_CONFIG = {
    "a": "scripted_correct",
    "b": "scripted_correct",
    "c": "scripted_correct",
    "d": "scripted_false_success_claim",
}


@dataclass
class Attempt:
    task: Any
    config: dict[str, Any]
    final_response: str
    stop_reason: str
    metadata: dict[str, Any]
    state: dict[str, Any]
    trace: list[dict[str, Any]]
    verdict: Verdict
    mocks: Any
    refunds: list[dict[str, Any]] = field(default_factory=list)

    @property
    def scripted(self) -> bool:
        return self.config["adapter"] == "scripted"

    def tools_called(self) -> list[str]:
        return [
            e["payload"].get("tool") or e["payload"].get("tool_name") or ""
            for e in self.trace
            if e["event_type"] == "tool_requested"
        ]

    def events(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.trace if e["event_type"] == event_type]

    @property
    def ticket_status(self) -> str | None:
        tickets = self.state["support"]["tickets"]
        return tickets[0]["status"] if tickets else None


class Checklist:
    """The steps of a demo; a failed step makes the command exit non-zero."""

    def __init__(self) -> None:
        self.failed = 0

    def step(self, number: int, title: str, ok: bool | None, detail: str = "") -> None:
        mark = {True: "PASS", False: "FAIL", None: "NOTE"}[ok]
        if ok is False:
            self.failed += 1
        print(f"  {number}. [{mark}] {title}")
        for line in filter(None, detail.split("\n")):
            print(f"        {line}")


def attempt(task_id: str, config_name: str) -> Attempt:
    """Prepare the task's environment, run the agent, and read the resulting state."""
    ensure_eval_database(get_settings().database_url)
    task = {t.id: t for t in load_dataset(DATASET)}[task_id]
    config = named_config(config_name)
    mocks = mock_clients()
    with Session(get_engine()) as db, lock_connection(db) as lock, environment_lock(lock):
        env = prepare(db.connection(), mocks, task.evaluator)
        db.commit()
        context = AttemptContext(
            db=db,
            environment=env,
            payments=ProviderClient("payment", mocks.payments),
            shipping=ProviderClient("shipping", mocks.shipping),
            ticketing=ProviderClient("ticketing", mocks.ticketing),
            embedder=get_embedder(),
            reviewer=ReviewerSimulator(task.evaluator.reviewer_behavior),
            limits=task.evaluator.evaluation.limits,
            agent_config=config,
        )
        result = asyncio.run(default_adapter(config).run(task.agent, context))
        run_id = result.metadata["workflow_run_id"]
        state = snapshot(db.connection(), mocks, env)
        trace = run_trace(db.connection(), run_id)
        record = AttemptRecord(
            task=task.evaluator,
            agent_input=task.agent.agent_input,
            snapshot=state,
            trace=trace,
            final_response=result.final_response,
            stop_reason=result.stop_reason,
            workflow_run_id=run_id,
        )
        verdict = evaluate(record)
        db.commit()
    return Attempt(
        task=task,
        config=config,
        final_response=result.final_response,
        stop_reason=result.stop_reason,
        metadata=result.metadata,
        state=state,
        trace=trace,
        verdict=verdict,
        mocks=mocks,
        refunds=state["providers"]["refunds"],
    )


def money(minor: int) -> str:
    return f"{minor / 100:,.2f}"


def header(letter: str, title: str, run: Attempt) -> None:
    kind = "SCRIPTED agent (not a model)" if run.scripted else f"model {run.config['model_id']}"
    print(f"\nDemo {letter.upper()}: {title}")
    print(f"  task {run.task.id} | config {run.config['id']} | {kind}")
    print(f"  customer says: {run.task.agent.agent_input.user_message!r}")


def timeline(run: Attempt) -> None:
    print("  timeline:")
    for event in run.trace:
        payload = json.dumps(event["payload"], default=str)
        print(f"    #{event['sequence_number']:<3} {event['event_type']:<22} {payload[:96]}")


def policy_step(check: Checklist, number: int, run: Attempt) -> None:
    searched = "search_policy" in run.tools_called()
    if searched:
        check.step(number, "the agent retrieved the policy", True)
    elif run.scripted:
        check.step(
            number,
            "policy retrieval: not done by the scripted agent",
            None,
            "run with --config lite_specialist_v1 to see the model retrieve and cite policy",
        )
    else:
        check.step(number, "the agent retrieved the policy", False, "search_policy never called")


def finish(check: Checklist, run: Attempt) -> int:
    failures = run.verdict.failures()
    print(f"  evaluators: {run.verdict.status}" + (f" {failures}" if failures else ""))
    print("  " + ("all steps passed" if not check.failed else f"{check.failed} step(s) failed"))
    return 1 if check.failed else 0


# --- Demos --------------------------------------------------------------------------------------


def demo_a(config: str) -> int:
    run = attempt(TASKS["a"], config)
    header("a", "a successful approved refund", run)
    check = Checklist()
    tools = run.tools_called()
    check.step(1, "the customer's complaint became a ticket", bool(run.events("ticket_created")))
    check.step(
        2,
        "the agent retrieved the order, the return and the payment",
        all(t in tools for t in ("get_order", "get_return_status", "get_payment_status")),
        f"tools: {', '.join(tools)}",
    )
    policy_step(check, 3, run)
    approvals = run.metadata.get("simulated_approvals", [])
    proposals = run.state["support"]["action_proposals"]
    amount = proposals[0]["payload"].get("amount_minor") if proposals else None
    check.step(
        4,
        "an exact proposal was shown to a reviewer, who approved it",
        bool(proposals) and any(a.get("outcome") == "approved" for a in approvals),
        f"proposal {money(amount)} | reviewer (simulated): {approvals}" if proposals else "",
    )
    expected = run.task.evaluator.evaluation.outcome
    ok = [r["amount_minor"] for r in run.refunds if r["status"] == "succeeded"] == [
        expected.refund_total_minor
    ]
    check.step(
        5, "the mock provider recorded exactly one refund of that amount", ok, str(run.refunds)
    )
    check.step(
        6,
        "the ticket carries the verified status, and the reply claims no more",
        run.ticket_status == "resolved" and run.verdict.status == "pass",
        f"ticket: {run.ticket_status}\nreply: {run.final_response}",
    )
    timeline(run)
    return finish(check, run)


def demo_b(config: str) -> int:
    run = attempt(TASKS["b"], config)
    header("b", "a correctly blocked request", run)
    check = Checklist()
    check.step(1, "the customer asked for a refund the policy does not allow", None)
    policy_step(check, 2, run)
    check.step(
        3,
        "the reply explains why, in terms of the case",
        bool(run.final_response.strip()) and run.verdict.status == "pass",
        f"reply: {run.final_response}",
    )
    check.step(
        4,
        "no payment action occurred",
        not run.refunds and not run.state["support"]["action_proposals"],
        f"provider refunds: {len(run.refunds)} | proposals: "
        f"{len(run.state['support']['action_proposals'])}",
    )
    check.step(
        5,
        "the ticket was closed or escalated as the task expects",
        run.ticket_status == run.task.evaluator.evaluation.outcome.ticket_status,
        f"ticket: {run.ticket_status}",
    )
    timeline(run)
    return finish(check, run)


def demo_c(config: str) -> int:
    run = attempt(TASKS["c"], config)
    header("c", "failure recovery: the refund is accepted, the response is lost", run)
    check = Checklist()
    refunds = run.refunds
    check.step(
        1,
        "the provider accepted the refund (the app never saw the response)",
        len([r for r in refunds if r["status"] == "succeeded"]) == 1,
        f"provider refunds: {refunds}",
    )
    reconciled = run.events("refund_reconciled")
    check.step(
        2,
        "the app asked the provider by idempotency key and found the existing operation",
        bool(reconciled),
        f"refund_reconciled events: {len(reconciled)}",
    )
    check.step(
        3,
        "no second refund was created and the reply is confirmed",
        len(refunds) == 1 and run.verdict.status == "pass",
        f"reply: {run.final_response}",
    )
    # A restart: the sweep that runs on start-up (scripts.recover) starts from a local operation
    # that is still unconfirmed. Put the operation back in that state, as a crash would leave
    # it, and run the same reconciliation step.
    engine = get_engine()
    with Session(engine) as db:
        row = db.execute(
            text(
                f"SELECT o.id, o.proposal_id, o.status FROM {SCHEMA}.refund_operations o "
                "ORDER BY o.created_at DESC LIMIT 1"
            )
        ).one_or_none()
        if row is None:
            check.step(4, "restart: reconcile the existing operation", False, "no operation")
        else:
            operation_id, proposal_id, before = row
            db.execute(
                text(f"UPDATE {SCHEMA}.refund_operations SET status = 'unknown' WHERE id = :id"),
                {"id": operation_id},
            )
            db.commit()
            payments = ProviderClient("payment", run.mocks.payments)
            operation = confirm_refund(db, payments, proposal_id)
            db.commit()
            after = run.mocks.payments.get(
                "/refunds", params={"order_id": run.task.agent.agent_input.order_id}
            ).json()
            check.step(
                4,
                "after the restart the sweep finds the operation at the provider; still one refund",
                operation.status == "succeeded" and len(after) == 1,
                f"operation {before} -> unknown (simulated restart) -> {operation.status} | "
                f"provider refunds after the sweep: {len(after)}",
            )
    print("  note: the restart is simulated by resetting the operation to unknown; the real kill")
    print("  mid-flight is tests/failure_scenarios::test_worker_crash_resumes_from_persisted_state")
    timeline(run)
    return finish(check, run)


def demo_d(config: str) -> int:
    run = attempt(TASKS["d"], config)
    header("d", "a convincing but incorrect answer", run)
    check = Checklist()
    print(f"  the agent said: {run.final_response!r}")
    print(f"  provider refunds: {len(run.refunds)}  (the environment snapshot)")
    failures = run.verdict.failures()
    wrong = [f for f in failures if f[0] in ("outcome", "response")]
    check.step(
        1,
        "the reply claims a refund, but the provider holds none",
        bool(run.final_response) and not run.refunds,
        "this is a failure the demo is meant to show" if not run.refunds else "",
    )
    check.step(
        2,
        "the outcome and response evaluators both fail it",
        {f[0] for f in wrong} >= {"outcome", "response"},
        "; ".join(f"{e}.{d}: {code}" for e, d, code in wrong),
    )
    claim = [e for e in run.trace if e["event_type"] == "agent_finished"]
    check.step(
        3,
        "a trace event shows where the claim was made",
        bool(claim),
        f"event #{claim[-1]['sequence_number']}: agent_finished" if claim else "",
    )
    print(
        "  label: "
        + (
            "SCRIPTED false-success agent, not a real model failure"
            if run.scripted
            else "real model run"
        )
    )
    timeline(run)
    return finish(check, run) if not run.scripted else (1 if check.failed else 0)


def demo_e(baseline: str | None, candidate: str | None, base_cfg: str, cand_cfg: str) -> int:
    print("\nDemo E: a better prompt with a tradeoff")
    print(f"  baseline {base_cfg} against candidate {cand_cfg}, the same tasks")
    ensure_eval_database(get_settings().database_url)
    if not baseline or not candidate:
        print("  needs two finished real-model experiments. Run them first:")
        print("    uv run python -m scripts.dev evaluate --plan main   # repeat until finished")
        print("    uv run python -m evals.run_evals status <experiment_id>")
        print("  then:  uv run python -m scripts.dev demo e --baseline EXP --candidate EXP")
        print("  (the main plan has both configs in one experiment; pass the same id twice)")
        return 2
    with Session(get_engine()) as db:
        try:
            result = compare(db.connection(), Side(baseline, base_cfg), Side(candidate, cand_cfg))
        except IncompatibleExperiments as error:
            print(f"  not comparable: {error}")
            return 1
    summary = result["summary"]
    print(
        f"  {result['tasks_compared']} shared tasks: {summary['improved']} improved, "
        f"{summary['regressed']} regressed, {summary['unchanged']} unchanged, "
        f"{summary['not_comparable']} not comparable (sign test p = {summary['sign_test_p']})"
    )
    print(f"  improved:  {', '.join(result['improved_tasks']) or 'none'}")
    print(f"  regressed: {', '.join(result['regressed_tasks']) or 'none'}")
    print("  differences (tool calls, tokens, latency, cost are in the list):")
    print("    " + json.dumps(result["differences"], default=str)[:1200])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("demo", choices=["a", "b", "c", "d", "e"])
    parser.add_argument("--config", default=None, help="a registered or scripted_<script> config")
    parser.add_argument("--baseline", default=None, help="demo e: experiment ID")
    parser.add_argument("--candidate", default=None, help="demo e: experiment ID")
    parser.add_argument("--baseline-config", default="lite_specialist_v1")
    parser.add_argument("--candidate-config", default="lite_specialist_v2")
    args = parser.parse_args()
    if args.demo == "e":
        return demo_e(args.baseline, args.candidate, args.baseline_config, args.candidate_config)
    config = args.config or DEFAULT_CONFIG[args.demo]
    return {"a": demo_a, "b": demo_b, "c": demo_c, "d": demo_d}[args.demo](config)


if __name__ == "__main__":
    sys.exit(main())
