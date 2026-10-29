# Customer Support Resolution Agent

An AI agent that takes a customer complaint from *"What happened?"* to *"Here's the verified outcome"*, using business data, support policies, APIs and human approval.

- **Stack:** Python · LangGraph · FastAPI · PostgreSQL (pgvector optional)
- **Core rule:** the model *proposes* actions. The application decides whether they are permitted and executes them through controlled tools.

## Scope and boundaries

Build for a **fictional online electronics retailer**. Supported complaint types:

| Type | Required behavior |
|---|---|
| **A. Order status** | Retrieve the order and shipment status, explain the latest available information, update the ticket |
| **B. Refund request** | Check eligibility, prepare a refund proposal, obtain approval, execute the approved refund, verify the result |
| **C. Damaged item** | Retrieve the order and relevant policy, collect required information, prepare a resolution or escalate |

First-version boundaries:

- One store and one currency.
- Human approval for **every** refund.
- Mock shipping and payment services.
- One customer issue per workflow.
- External customer messaging stays inside the demo interface.
- Excluded: chargebacks, fraud adjudication, complex partial returns.
- Synthetic customer data and a mock payment service throughout.

The reference interaction the system must handle end to end:

> *"I returned my headphones last week, but I still haven't received my refund."*
>
> 1. Identify the authenticated customer
> 2. Retrieve the order, return and payment records
> 3. Find the applicable refund policy
> 4. Determine whether the refund is eligible, already processing, or blocked
> 5. Propose an appropriate action
> 6. Request human approval before issuing a new refund
> 7. Call the refund API after approval and validation
> 8. Update the support ticket
> 9. Explain what happened without claiming more than the API confirmed

## Setup

Prerequisites: Python 3.13, [uv](https://docs.astral.sh/uv/), and Docker with Compose. Node.js is only needed to render the diagrams.

The short path. Each command also works as `make <command>`, or as `uv run python -m scripts.dev <command>` on Windows without make:

```bash
docker compose up -d postgres               # PostgreSQL
docker compose --profile mocks up -d        # mock payment, shipping and ticketing services (8001-8003)
uv run python -m scripts.dev setup          # locked dependencies, .env from .env.example, the support_eval database
uv run python -m scripts.dev migrate        # both databases: the support schema and the eval schema
uv run python -m scripts.dev seed           # demo business data (resets business tables), policies, dataset, scripted configs
uv run python -m scripts.dev smoke          # 5 pilot tasks with the scripted agent: no API key needed
```

`smoke` is the quick proof that the platform works: it must print `5  completed / pass`. Two databases are used on purpose. `support` is the application database. `support_eval` is the experiment database, because every evaluation attempt empties the business tables. The `smoke`, `evaluate`, `report` and `demo` commands always point at `support_eval`, and the experiment worker refuses any database whose name does not end in `_eval` or `_test`.

| Command | What it does |
|---|---|
| `setup` | Install locked dependencies, create `.env`, create the `support_eval` database |
| `migrate` | Apply all migrations to both databases |
| `seed` | Seed business data and the mocks (`--no-mocks` to skip them), ingest policies, import `support-v1` and the scripted configs |
| `test` | The deterministic test suite (no model calls; extra arguments go to pytest) |
| `smoke` | The 5 pilot tasks with the scripted adapter |
| `evaluate --plan NAME` | Create a real-model experiment from `evals/plans/NAME.yaml` and run it |
| `report EXPERIMENT_ID` | Export a finished experiment to `evals/reports/<id>/` |
| `demo a`, `b`, `c`, `d`, `e` | The demo scenarios, see [docs/demo-scenarios.md](docs/demo-scenarios.md) |

More commands:

```bash
uv run uvicorn app.main:app --reload     # the agent API on http://localhost:8000 (GET /health, /demo, /review)
uv run python -m scripts.mint_token auth|alice                   # a demo JWT (--role reviewer|senior_reviewer|operator)
uv run python -m app.jobs.worker         # job worker, for RUN_EXECUTION=worker (202 Accepted + polling)
uv run python -m scripts.recover         # reconcile uncertain refunds and resume unfinished runs
uv run uvicorn evals.api:app --port 8100 # the evaluation API and screens (/ui/experiments), operator token
uv run python -m scripts.export_diagrams --render   # diagrams from hld.md into docs/diagrams/
uv run ruff check .                      # lint
docker compose down                      # stop containers (data is kept); `down -v` also deletes the database
```

### Configuration

Every variable is in `.env.example`; none is a real secret.

- **LLM:** Gemini API, free tier, no billing linked. Put a free Google AI Studio key in `LLM_API_KEY` and a model ID in `LLM_MODEL`. The experiments use `gemini-3.5-flash-lite` as the main model and `gemini-3.5-flash` for one comparison. The experiment worker waits out the per-minute request limit, and defers jobs to the next day when the daily quota runs out instead of failing them.
- **Embeddings:** local sentence-transformers `all-MiniLM-L6-v2` (384 dimensions) stored with pgvector. No API key. The first policy ingestion downloads about 90 MB.
- **Run execution:** `RUN_EXECUTION=inline` (the request drives the run) or `worker` (the API returns `202 Accepted` and a worker drives it).
- **Money and approval:** amounts are integer minor units (paise), one currency (`STORE_CURRENCY=INR`), and a pending approval expires after `APPROVAL_EXPIRY_MINUTES` (60).
- **Mocks:** `MOCK_*_URL` and `MOCK_*_API_KEY` for the three services, and `PROVIDER_TIMEOUT_SECONDS` (5). The evaluation clients use the same timeout, so a timeout fault really looks like a timeout.

## How it works

The architecture diagrams are in [docs/diagrams/](docs/diagrams/README.md), exported from [hld.md](hld.md). In one paragraph: a complaint becomes a ticket. A LangGraph workflow classifies it, gathers facts and policy, and asks the model for a structured proposal. The proposal is validated against business rules in code, the run pauses for human approval, and ordinary code then executes the refund, verifies it and updates the ticket.

**Who decides what.** The model classifies, summarizes evidence, interprets policy and proposes. Code decides identity, ownership, amounts, eligibility, approval validity, execution, retries and state transitions. Customer identity comes from the authenticated request (a JWT mapped through `customers.external_auth_subject`). The model never chooses the customer, the reviewer, the approval status or any payment credential.

**Tools.** Read tools: `get_order`, `list_orders`, `get_shipment_status`, `get_return_status`, `get_payment_status`, `get_refund_status`, `get_ticket`, `search_policy`. Write tools: `create_refund_proposal` (records a proposal and moves no money), `execute_approved_refund` and `update_ticket`. Every tool has typed inputs and a structured result (`success`, `error_code`, `retryable`, `message`). Ownership is checked inside each service, so an order ID supplied by the customer or the model grants nothing. Each agent gets only the tools it needs: the investigation agent has no write tool, the policy agent has `search_policy` only, and execution is reachable only through the proposal step and the code-owned execute step. A single-agent baseline exists next to the specialist design, so the two can be compared.

**Approval design.** Every refund needs human approval, bound to one immutable proposal (a stored payload and its hash). Words in a chat message authorize nothing. The reviewer is authenticated and role-checked (a `senior_reviewer` sees approvals that a standard reviewer does not). The run pauses with LangGraph `interrupt()`. On resume the code re-checks expiry, the payload hash, eligibility and the refundable balance. An expired approval, or one whose order, amount or material details changed, needs a new approval. Nothing with a financial effect happens before the interrupt.

**Refund idempotency and recovery.** A refund operation reserves its amount in a transaction (`SELECT ... FOR UPDATE`) and has a stable idempotency key with a unique constraint. Successful refunds plus pending reservations can never exceed the refundable balance, and no transaction is held during the network call. The mock payment API returns the same operation for the same key and payload, and rejects the same key with a different payload. If the provider accepts a refund but the response is lost, the operation becomes `unknown`, and the code asks the provider by idempotency key before it retries or tells the customer anything. After a crash or restart, `scripts.recover` (which the worker also runs on start) reconciles uncertain operations first, then resumes unfinished runs from their LangGraph checkpoints. If the refund succeeds but the ticket update fails, only the ticket update is retried. The customer is told no more than the provider confirmed.

**Policies (RAG).** Synthetic policy documents have an ID, a version and effective dates, and are split by section. `search_policy` filters to the version that applies to the order and returns citations such as `refund_policy_v1:section_3`. It reports missing or conflicting evidence, escalates if no policy applies, and treats text inside retrieved passages as untrusted. A similarity score never proves eligibility: financial conditions are enforced in code.

## Evaluation platform

The platform asks one question: *if I change the agent's model, prompt, tools or orchestration, does it become more reliable, and at what cost?* The resolution agent is the system under test. Design notes are in [hld.md](hld.md) section 10, and the task list is in [tasks.md](tasks.md).

- **Tasks and the two views.** A task is a YAML file in `evals/datasets/support-v1/tasks/` with an *agent view* (the customer, the order and the message: all the agent ever receives) and an *evaluator view* (the fixture, the expected outcome, the allowed and forbidden actions, the rubric). The agent never receives expected outcomes, hidden fixtures or grading rules, and the run screen shows the two views apart. The dataset `support-v1` has 50 versioned, hashed tasks, split into dev (30), validation (10) and a held-out test split (10). The test split needs `--allow-test-split` and is read only after prompts are frozen. There is also a 5-task pilot.
- **A clean environment per attempt.** Each attempt resets the business tables and the three mock services, loads the task's fixture and arms its faults (errors, delays, "accept the refund but time out"). It takes a final-state snapshot before the next reset.
- **Simulated approvals.** A deterministic reviewer simulator approves, rejects or lets an approval expire, as the task says. It goes through the real approval service and is recorded as simulated. The agent never sees its behavior.
- **Four evaluators, deterministic only.** *Outcome* (provider refunds, amounts, ticket and run status, read from the snapshot), *policy* (no refund without valid approval, no cross-customer access, no duplicate or over-refund, untrusted content), *trajectory* (required steps and preconditions, schema validity, argument correctness, redundant or irrelevant calls, failure handling) and *response* (must mention, must not claim, false refund claims). An LLM judge may later assist with wording only. It never decides whether money moved correctly. A blocked unsafe attempt is reported separately from an executed violation.
- **Metrics, each with its denominator.** Verified success (successful ÷ scorable runs), strict end-to-end success (÷ all planned runs), coverage (scorable ÷ planned), unsafe attempt rate, executed violation rate, duplicate refunds at the provider, tool schema validity, tool-call precision, required-call recall, policy retrieval quality, latency (queue, active and reviewer wait, separately), and tokens and estimated cost per successful run. Fully automated resolutions are reported apart from human-assisted ones. Repeated trials report the mean, per-task variability, "passed all repetitions" and task-level bootstrap intervals. Missing token usage is *unknown*, never free, and cost is an estimate from price versions.
- **Comparison.** Only compatible experiments are compared (same dataset version, split, environment, evaluator versions and limits). The result names the tasks that improved and the **regressed tasks**, with differences in success, coverage, tool calls, tokens, cost and latency.
- **Reproducibility.** Every experiment saves a manifest: dataset hash, configs, model IDs as resolved by the provider, prompt hashes, tool-schema hash, code commit, lockfile hash, faults, and environment and evaluator versions. `report` exports it with the scorecard and one row per run as JSON and CSV, and `python -m evals.run_evals rerun <id>` repeats an experiment from its manifest.
- **API and screens.** An operator-only `/eval` API (`uvicorn evals.api:app`) and Jinja2 screens: the experiments list, one experiment (scorecard, categories, failure reasons, latency, runs), one run (agent-visible input apart from ground truth, evaluator findings, final state against expected, trace) and a comparison. They show structured events, not private model reasoning.

### The four configurations

All four use the same tasks, environment, limits, reviewer behavior and faults.

| Config | Model | Architecture | Prompt |
|---|---|---|---|
| `lite_single_v1` | gemini-3.5-flash-lite | single agent | v1 |
| `lite_specialist_v1` | gemini-3.5-flash-lite | specialists | v1 |
| `lite_specialist_v2` | gemini-3.5-flash-lite | specialists | v2 (explicit investigation and approval instructions) |
| `flash_specialist_v1` | gemini-3.5-flash | specialists | v1 |

Three comparisons, each changing one thing: architecture (`lite_single_v1` vs `lite_specialist_v1`), prompt (`lite_specialist_v1` vs `lite_specialist_v2`) and model (`lite_specialist_v1` vs `flash_specialist_v1`, on the validation split only, because Flash allows few requests a day). The run plans are in `evals/plans/`.

## Measured results

Nothing here is production performance. Every number comes from synthetic tasks against mock services, and each states its dataset, metric, sample size and conditions.

| What | Dataset and split | Config | Metric | Result | Sample |
|---|---|---|---|---|---|
| Pilot, 2026-10-29 | `support-v1`, pilot | `lite_specialist_v1` | runs scored, evaluation outcome, model calls per run | 5 of 5 scored; 4 pass, 1 partial; 6.4 calls per run | 5 tasks, 1 repetition |

This is a smoke-sized pilot, not evidence about the agent. The main experiment (150 runs), the repetitions, the model comparison and the held-out test split are in progress. They are added here, with their report files, when they finish. Until then the README makes no claim about which configuration is better.

## Known limitations and planned extensions

- **Synthetic and small.** One fictional store, one currency, 50 hand-written tasks and mock providers. It says nothing about real traffic.
- **The scripted agent is not a model.** It exists to prove the platform and to show failures. It does not retrieve policy.
- **One attempt at a time** per environment (an advisory lock), so large experiments are slow and depend on the free-tier quota. Concurrent attempts are an extension.
- **Simulated approvals** replace a human reviewer in evaluation runs. The real approval flow is tested separately.
- **Deterministic evaluators only.** No LLM judge, no simulated multi-turn customer and no public benchmark (BFCL and τ²-bench are possible extensions).
- **Cost is an estimate** from registered paid-tier prices. Actual free-tier cost is zero, and runs without token usage are reported as unknown cost.
- **The recovery demo is partly simulated.** Demo C resets an operation to `unknown`. The real mid-flight kill is covered by a failure-scenario test.
- **Excluded by scope:** chargebacks, fraud adjudication, complex partial returns, and customer messaging outside the demo screens.

## Demo

The five scenarios and how to run each are in [docs/demo-scenarios.md](docs/demo-scenarios.md). A recording is not published yet.

## Testing

```bash
uv run python -m scripts.dev test     # the deterministic suite; PostgreSQL must be running
uv run pytest -m llm -s tests/evals/test_week4_gate_llm.py   # optional: real model, uses free-tier quota
```

Integration tests use a separate `support_test` database that they drop, create and migrate themselves each session.

## Status

Phase 0, Weeks 1 to 3 (agent, retrieval, approval, reliability) and Week 4 (evaluation platform) are done. Week 5 is in progress. Metrics, comparison, the API, report export, the screens, the developer commands, demo scenarios A to D and the diagrams are built. The real-model experiments are running, and Demo E, the recording and the résumé numbers wait for them. Progress per task is in [tasks.md](tasks.md).
