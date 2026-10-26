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

Prerequisites: Python 3.13, [uv](https://docs.astral.sh/uv/), and Docker with Compose.

```bash
# 1. Install dependencies from the lockfile (creates .venv)
uv sync

# 2. Create your local environment file, then edit it if needed
cp .env.example .env

# 3. Start PostgreSQL
docker compose up -d

# 4. Start PostgreSQL + mock payment, shipping and ticketing services
#    (the mock apps are added in Week 1, tasks W1.8, W1.9 and W1.10)
docker compose --profile mocks up -d
```

Useful commands:

```bash
uv run alembic upgrade head    # create or update the database schema
uv run alembic downgrade base  # remove the schema (keeps the database)
uv run uvicorn app.main:app --reload  # run the API on http://localhost:8000 (GET /health)
uv run python -m scripts.seed_data    # load the demo scenarios (resets business tables)
uv run python -m scripts.seed_data --push-mocks  # ...and load the mock payment data too
uv run python -m scripts.mint_token auth|alice   # demo JWT (--role reviewer|senior_reviewer)
docker compose --profile mocks up -d mock-payments  # mock payment API on http://localhost:8001
uv run pytest                  # run the tests (integration tests need PostgreSQL running)
uv run ruff check .            # lint
docker compose down            # stop containers (data is kept)
docker compose down -v         # stop containers and delete the database volume
```

Model and embedding configuration (see `.env.example`):

- **LLM:** Gemini API free tier with a Flash model. Put a free Google AI Studio key in `LLM_API_KEY`. `LLM_PROVIDER` can be switched to a local model if the free quota runs out.
- **Embeddings:** local sentence-transformers `all-MiniLM-L6-v2` (384 dimensions), stored with pgvector. No API key needed.

## Status

Phase 0 (repository setup). The application, workflow and mock services are built in Weeks 1–4.
