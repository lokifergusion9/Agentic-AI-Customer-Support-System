# Implementation Tasks — Customer Support Resolution Agent

Source: `Project.pdf`, *"A practical Agentic AI project for aspiring Forward Deployed Engineers"*. Architecture: [`high-level-diagram.md`](high-level-diagram.md).

- **Stack:** Python · LangGraph · FastAPI · PostgreSQL (pgvector optional)
- **Level:** Intermediate
- **Timeline:** 3–4 weeks
- **Goal:** an AI agent that takes a customer complaint from *"What happened?"* to *"Here's the verified outcome"*, using business data, support policies, APIs and human approval.

How to read this file:

- Each task has an ID (`P0.x`, `W1.x` … `W4.x`), a checkbox, and the PDF section it comes from (`§n`).
- Each phase ends with the PDF's **Done when** gate. Do not start the next phase until that gate passes.
- This is a plan only. No application code or dependencies are added at this stage.

---

## Scope and boundaries (copy into README — PDF §3)

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

The reference interaction the system must handle end to end (PDF §1):

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

---

## Phase 0 — Repository setup

- [x] **P0.1 — Create the repository skeleton** (§17). Use the exact layout:
  ```text
  customer-support-resolution-agent/
  ├── app/{main.py, api/, auth/, schemas/, agents/, workflows/, tools/, services/, policies/, db/}
  ├── mock_services/{payments/, shipping/}
  ├── migrations/
  ├── scripts/{seed_data.py, ingest_policies.py}
  ├── evals/{datasets/, run_evals.py, reports/}
  ├── tests/{unit/, integration/, failure_scenarios/}
  ├── docs/
  ├── docker-compose.yml
  ├── .env.example
  ├── pyproject.toml
  └── README.md
  ```
- [x] **P0.2 — Project config** (§17). Set up `pyproject.toml` with dependencies **pinned in a lockfile**.
- [x] **P0.3 — Local infra** (§5, §17). Add `docker-compose.yml` with PostgreSQL (plus the pgvector extension if that option is chosen) and the mock services.
- [x] **P0.4 — Environment template** (§17). `.env.example` lists every required variable: DB URL, model configuration, embedding configuration, mock service URLs, and approval expiry. No real secrets.
- [x] **P0.5 — README stub** (§3, §19). Write the scope and boundaries section above into the README before writing any code.

---

## Week 1 — Business workflow and backend (PDF §16)

**Done when:** you can manually resolve a sample ticket through the APIs **without an LLM**.

### Data and schema

- [ ] **W1.1 — Synthetic policy documents** (§9). Write a small set covering:
  - refund eligibility
  - return processing
  - damaged products
  - shipping delays
  - exceptions requiring human review

  Give each document an ID, a version and effective dates. Example policy from the PDF: *"Refunds for returned items become eligible after the return is received and inspection passes. A support reviewer must approve every refund."*
- [ ] **W1.2 — Synthetic customer records** (§1, §16). In `scripts/seed_data.py`, seed customers, orders, order items and returns. Include enough variety for the eval categories in W4.1: eligible, ineligible, already refunded, return not received, inspection failed, damaged, delayed shipment, and orders that belong to another customer.
- [ ] **W1.3 — PostgreSQL schema and migrations** (§7). One migration per table. Keep the PDF field set:

  | Table | Important fields |
  |---|---|
  | `customers` | `id`, `external_auth_subject`, `name`, `email` |
  | `orders` | `id`, `customer_id`, `status`, `currency`, `paid_amount_minor`, `delivered_at` |
  | `order_items` | `id`, `order_id`, `product_id`, `quantity`, `paid_amount_minor` |
  | `returns` | `id`, `order_id`, `status`, `received_at`, `inspection_status` |
  | `tickets` | `id`, `customer_id`, `order_id`, `category`, `status`, `summary` |
  | `workflow_runs` | `id`, `ticket_id`, `thread_id`, `status`, `started_at`, `completed_at` |
  | `policy_documents` | `id`, `version`, `effective_from`, `effective_to`, `source_uri` |
  | `policy_chunks` | `id`, `document_id`, `content`, `embedding`, `metadata` |
  | `action_proposals` | `id`, `run_id`, `action_type`, `payload`, `payload_hash`, `status` |
  | `approvals` | `id`, `proposal_id`, `reviewer_id`, `decision`, `expires_at`, `decided_at` |
  | `refund_operations` | `id`, `order_id`, `proposal_id`, `amount_minor`, `idempotency_key`, `provider_reference`, `status` |
  | `audit_events` | `id`, `run_id`, `actor`, `event_type`, `redacted_payload`, `created_at` |

- [ ] **W1.4 — Database rules** (§7):
  - [ ] Money is stored in **integer minor units** (paise / cents).
  - [ ] **Unique constraint** on `refund_operations.idempotency_key`.
  - [ ] Track successful refunds **and** amounts reserved by pending operations.
  - [ ] Total committed refunds can never exceed the refundable balance, enforced in a transaction.
  - [ ] Record approval timestamps and the **exact proposal** that was approved.
  - [ ] Migrations reproduce the schema from scratch for another developer.
  - [ ] Application tables are kept logically separate from LangGraph checkpoint tables (§5).

### Auth and API

- [ ] **W1.5 — Customer authentication** (§5, §8). Customer identity is derived from the authenticated request and mapped through `customers.external_auth_subject`. Nothing downstream accepts a customer ID from the request body or from the model.
- [ ] **W1.6 — Reviewer authentication and authorization** (§11). Add a separate reviewer role and permission check, used later by the approval endpoints.
- [ ] **W1.7 — Ticket endpoints** (§13):
  - [ ] `POST /tickets`: create a customer support ticket (ownership-checked `order_id`).
  - [ ] `GET /health`: check application health.
  - [ ] Authorization is checked on every endpoint, with input and response validation.

### Mock external services

- [ ] **W1.8 — Mock payment / refund API** (§5, §12):
  - [ ] Refund request accepts an idempotency key. The **same key and same payload return the same operation**. The **same key with a changed payload is rejected**.
  - [ ] Payment status and refund status can be queried by order, by key, or by provider reference.
  - [ ] Fault injection: errors, delayed results, and *"accept the refund but time out before responding"*.
- [ ] **W1.9 — Mock shipping API** (§5). Shipment status for order-status complaints, with error and delay injection.
- [ ] **W1.10 — Mock ticketing API** (§5). Ticket status and notes updates, with failure injection so the *"refund succeeds, ticket update fails"* case can be tested. Lives in `mock_services/ticketing/` (Q1 decided).
- [ ] **W1.11 — Mock order API** (§16). Week 1 lists *"mock order, shipping, and payment APIs"*. Orders are read through a service layer over the seeded tables, or a thin mock over them.

### Deterministic business logic

- [ ] **W1.12 — Read services** (§8). Each one checks ownership **inside the service**. An order ID supplied by a customer or the model does not grant access. Each returns typed, structured results:
  - `get_order(order_id)`
  - `get_return_status(order_id)`
  - `get_payment_status(order_id)`
  - `get_refund_status(operation_id)`
  - `get_ticket(ticket_id)`
- [ ] **W1.13 — Structured failures** (§8). Every service returns failures in this shape:
  ```json
  { "success": false, "error_code": "RETURN_NOT_RECEIVED", "retryable": false, "message": "The return has not been marked as received." }
  ```
- [ ] **W1.14 — Deterministic refund eligibility** (§6, §16). Pure code that decides *eligible / already processing / blocked*. It covers the return-received check, the inspection status, and the refundable balance (paid amount minus successful refunds minus pending reservations), all in minor units.
- [ ] **W1.15 — Manual resolution path** (§16). Through the APIs and services alone (no LLM):
  1. create a ticket
  2. check eligibility
  3. create a proposal
  4. approve it
  5. execute the mock refund
  6. update the ticket

  Write this as an integration test in `tests/integration/`.

**Gate check:** W1.15 passes for a sample refund ticket and a sample order-status ticket.

---

## Week 2 — Agent and retrieval (PDF §16)

**Done when:** the system gathers evidence and produces a **validated proposal with citations**.

### RAG for support policies

- [ ] **W2.1 — Policy ingestion pipeline** (§9). In `scripts/ingest_policies.py`:
  1. Load the policy docs from W1.1, with their document IDs, versions and effective dates.
  2. Split them into meaningful sections.
  3. Generate embeddings.
  4. Store chunks and metadata in `policy_documents` / `policy_chunks`. pgvector is optional (§5).
- [ ] **W2.2 — `search_policy(query, policy_filters)`** (§8, §9):
  - [ ] Retrieve passages using the complaint **plus the relevant order facts**.
  - [ ] Filter to the policy version that applies (effective dates vs. order or delivery date).
  - [ ] Preserve document and section references, and return citations such as `refund_policy_v1:section_3`.
  - [ ] Report when evidence is missing or conflicting.
  - [ ] Escalate if no applicable policy can be established.
  - [ ] Treat instructions embedded in retrieved text as **untrusted content**.
  - [ ] Never use a similarity score as proof that a refund is valid. Financial conditions are enforced in code.
- [ ] **W2.3 — Retrieval routing rule** (§9). Policy questions use retrieval, for example *"What is the return window?"*. Current facts use APIs or SQL, for example *"When was this order delivered?"* (order API) and *"Has this payment already been refunded?"* (payment API).

### Typed tools

- [ ] **W2.4 — Wrap the services as narrow, typed tools** (§8), with typed inputs and structured outputs.
  - Read: `get_order(order_id)`, `get_return_status(order_id)`, `get_payment_status(order_id)`, `get_refund_status(operation_id)`, `search_policy(query, policy_filters)`, `get_ticket(ticket_id)`.
  - Write: `create_refund_proposal(order_id, amount_minor, reason)`, `execute_approved_refund(proposal_id)`, `update_ticket(ticket_id, status, note)`.
- [ ] **W2.5 — Tool safety rules** (§8):
  - [ ] Identity is injected from the authenticated request, never from tool arguments.
  - [ ] Privileged parameters stay under server control. The model never supplies reviewer identity, approval status or payment credentials.
  - [ ] `create_refund_proposal` **never issues money**. Proposal and execution stay separate.

### Workflow state and graph

- [ ] **W2.6 — Structured workflow state** (§10), with enough state to inspect and resume the workflow:
  ```text
  run_id, ticket_id, authenticated_customer_id, order_id, complaint_category,
  customer_message, verified_facts, policy_references, missing_information,
  proposed_action_id, approval_id, execution_status, external_result_reference,
  retry_counts, error_details, final_response
  ```
- [ ] **W2.7 — Validated schemas for specialist outputs** (§10). The resolution proposal must match:
  ```json
  {
    "action": "request_refund_approval",
    "order_id": "ord_1042",
    "amount_minor": 249900,
    "currency": "INR",
    "reason": "Return received and inspection passed",
    "evidence_refs": ["return_ret_42", "payment_pay_92"],
    "policy_refs": ["refund_policy_v1:section_3"],
    "missing_information": []
  }
  ```
  The application validates the amount, ownership and eligibility **independently** of these values.
- [ ] **W2.8 — Complaint classification** (§4, §16). Classify into `order_status`, `refund_request` or `damaged_item`. Anything else is out of scope and is escalated.
- [ ] **W2.9 — Single-agent baseline first** (§6). Build one tool-using agent with **deterministic control flow** that runs the §4 flow end to end. Keep it runnable, because W4.4 compares against it.
- [ ] **W2.10 — Specialist agents** (§6), each with **only the tools it needs**:

  | Agent | Responsibility | Output | Tools |
  |---|---|---|---|
  | Investigation | Gather customer, order, return and payment facts | Structured evidence summary | read tools only, **no refund execution** |
  | Policy | Retrieve applicable policy passages | Citations and policy interpretation | `search_policy` |
  | Resolution | Combine facts and policy into a proposed action | Structured resolution proposal | `create_refund_proposal` |

  - [ ] Run investigation and policy retrieval **concurrently** when they are independent, then combine their outputs.
  - [ ] When policy selection depends on the product or order date, gather facts first.
- [ ] **W2.11 — LangGraph routing** (§4, §5):
  - authenticate & validate
  - classify
  - retrieve facts and policies
  - check for missing information
  - create proposal
  - **validate against business rules (code)**
  - route to **answer only**, **request approval** or **escalate to human**
- [ ] **W2.12 — Ordinary code, not the LLM, for** (§6): authentication, access control, amount calculations, eligibility enforcement, approval validation, API execution, retry limits and state transitions.
- [ ] **W2.13 — Missing-information loop** (§4, §14). Ask a targeted clarification question, pause, and resume when `POST /runs/{run_id}/messages` supplies the additional information.
- [ ] **W2.14 — Run endpoints** (§13):
  - [ ] `POST /tickets/{ticket_id}/runs`: start a resolution workflow.
  - [ ] `GET /runs/{run_id}`: retrieve status and output.
  - [ ] `GET /runs/{run_id}/events`: retrieve a **redacted** event timeline.
  - [ ] `POST /runs/{run_id}/messages`: supply additional customer information.
- [ ] **W2.15 — Model-output guardrails** (§14). On invalid model output: validate, attempt a limited repair, then stop or escalate. Stop at a configured **step or cost limit** to prevent excessive workflow loops.

**Gate check:** the headphones refund example produces a schema-valid proposal with evidence refs and policy citations, and the business-rule validator accepts it.

---

## Week 3 — Approval and reliability (PDF §16)

**Done when:** restarting or retrying a run **does not issue duplicate refunds**.

### Human approval

- [ ] **W3.1 — Immutable proposals** (§11). Persist the proposal to `action_proposals` with its `payload_hash`. It is never edited after creation.
- [ ] **W3.2 — Pending approval bound to that proposal** (§11). Create a row in `approvals` with `expires_at`.
- [ ] **W3.3 — Durable pause/resume** (§11). Use LangGraph `interrupt()`, a checkpointer (PostgreSQL, in separate checkpoint tables), and `Command(resume=...)` with the **same thread ID**. The interrupted node restarts on resume, so **no financial side effect may happen before `interrupt()`**. Keep the side effects in a separate execution step.
- [ ] **W3.4 — Approval endpoints** (§13):
  - [ ] `GET /approvals`: list approvals available to the authenticated reviewer.
  - [ ] `POST /approvals/{approval_id}/decision`: approve or reject. The reviewer identity comes from auth. The decision and `decided_at` are saved, then the run is resumed.
- [ ] **W3.5 — Approval lifecycle** (§11), in this order:
  1. Persist an immutable proposal.
  2. Create a pending approval bound to that proposal.
  3. Pause the workflow.
  4. Authenticate the reviewer.
  5. Validate reviewer permissions.
  6. Save the decision.
  7. Resume the same workflow.
  8. Recheck eligibility and refundable balance.
  9. Execute **only the approved action**.
- [ ] **W3.6 — Invalidation** (§11, §14). If the order, the amount or other material details change, invalidate the old approval and request a new one. An expired approval requires a new valid approval. A rejected approval records the decision and follows the rejection path.
- [ ] **W3.7 — Chat is not authorization** (§11). The execution service verifies the **stored** approval and the authenticated reviewer decision. It never accepts an approval message from the chat.
- [ ] **W3.8 — Reviewer interface** (§11, §16). The screen shows:
  - customer and order
  - complaint summary
  - proposed action and amount
  - relevant evidence
  - policy citations
  - approve and reject controls
  - proposal expiration time

### Idempotent refunds

- [ ] **W3.9 — Refund execution pattern** (§12):
  1. Create a durable refund operation with a **stable idempotency key**.
  2. In a short DB transaction, validate the approval and **reserve** the refundable amount (`SELECT ... FOR UPDATE`).
  3. **Commit before** calling the external service. Never hold the transaction open during the network call.
  4. Call the provider with the stable key.
  5. Persist its response.
  6. If the result is uncertain, mark the operation `unknown` or `pending_reconciliation`.
  7. Query the provider by key or reference before deciding the next step.
  8. Retry using the **same key** only when the provider contract supports it.
- [ ] **W3.10 — Concurrency control** (§12). Two concurrent requests for the same order can never over-reserve the refundable balance.
- [ ] **W3.11 — Verify the external result** (§4). The run never reports success until the provider result is confirmed.

### Recovery, tickets and audit

- [ ] **W3.12 — Reconciliation** (§12, §14). On restart, find runs and operations in an uncertain state and reconcile them with the provider **before** retrying or claiming success.
- [ ] **W3.13 — Job execution model** (§13). The prototype runs workflows inline. The robust version:
  - persists a job
  - returns `202 Accepted` with the run ID
  - executes on a worker (FastAPI `BackgroundTasks` or Celery)
  - persists progress
  - lets clients query status

  Checkpointing restores state, but a mechanism is still needed to schedule and restart unfinished jobs.
- [ ] **W3.14 — Ticket updates** (§4, §14). `update_ticket` reflects the **verified** external status. If the refund succeeded but the ticket update failed, retry **only the ticket update**.
- [ ] **W3.15 — Audit trail** (§7, §13). Write an `audit_events` row for every state transition, tool call, decision and external result. The payload is **redacted**. `GET /runs/{run_id}/events` serves it.
- [ ] **W3.16 — Failure-scenario tests** (§14). Add one test per row in `tests/failure_scenarios/`:

  | Failure | Expected behavior |
  |---|---|
  | Missing order information | Ask a targeted clarification question |
  | Order belongs to another customer | Deny access without revealing its details |
  | No applicable policy found | Escalate rather than invent eligibility |
  | Conflicting policy evidence | Flag the conflict for review |
  | Temporary read API failure | Retry with a bounded backoff policy |
  | Invalid model output | Validate, attempt limited repair, then stop or escalate |
  | Approval rejected | Record the decision and follow the rejection path |
  | Approval expired | Require a new valid approval |
  | Refund outcome uncertain | Reconcile before retrying or claiming success |
  | Refund succeeds, ticket update fails | Retry only the ticket update |
  | Worker crashes | Resume from persisted state and reconcile side effects |
  | Excessive workflow loops | Stop at a configured step or cost limit |

**Gate check:** kill the worker after the provider accepts a refund but before the response is persisted, then restart. Exactly one refund exists at the provider, and the ticket shows the verified status.

---

## Week 4 — Evaluation and presentation (PDF §16)

**Done when:** another developer can run the project and **reproduce your reported results**.

### Evaluation

- [ ] **W4.1 — Synthetic eval dataset, about 60 cases** (§15), in `evals/datasets/`:

  | Category | Cases |
  |---|---|
  | Order status and shipment questions | 10 |
  | Eligible refund requests | 10 |
  | Ineligible refund requests | 10 |
  | Missing information and ambiguous complaints | 10 |
  | API failures, retries, and crashes | 10 |
  | Authorization, concurrency, and adversarial inputs | 10 |

  - [ ] Keep a **held-out subset** that is never used while tuning prompts.
  - [ ] Store an expected outcome for each case:
    - relevant customer and order
    - applicable policy
    - allowed actions
    - forbidden actions
    - whether approval is required
    - expected final business state
    - expected escalation or clarification behavior
- [ ] **W4.2 — Eval harness** (§15). `evals/run_evals.py` seeds a scenario, runs the workflow, supplies any simulated reviewer decision, and **checks the resulting system state**. It uses DB assertions and mock-provider records for transactional outcomes. An LLM judge may assist with response quality only. It never decides whether money moved correctly.
- [ ] **W4.3 — Metrics** (§15), written to `evals/reports/`:
  - [ ] **Verified resolution rate:** cases whose expected business outcome is confirmed ÷ cases designated resolvable.
  - [ ] **Overall workflow correctness:** cases handled according to their expected outcome (including correct clarification, refusal and escalation) ÷ all cases.
  - [ ] **Unauthorized action count:** refunds or data accesses performed without the required authorization.
  - [ ] **Duplicate refund count:** duplicate financial effects under retries, restarts and concurrent execution.
  - [ ] **Policy retrieval quality:** whether the required policy evidence appears in the retrieved results.
  - [ ] **Grounded response quality:** whether factual claims are supported by retrieved records and tool results.
  - [ ] **Operational metrics:** latency, model calls, tool calls, retry count, and estimated cost per case. Measure active execution time separately from time spent waiting for a reviewer.
  - [ ] Report **human-assisted** resolution separately from **fully automated** resolution.
- [ ] **W4.4 — Baseline comparison** (§6, §16). Compare the specialist design (W2.10) against the single-agent baseline (W2.9). Check whether the extra coordination improves correctness enough to justify its cost and latency.

### Demo and documentation

- [ ] **W4.5 — Simple demo interface** (§18). It only needs:
  - customer complaint panel
  - workflow timeline
  - evidence and policy panel
  - approval queue
  - final outcome panel

  Display structured events and concise explanations, **not private model reasoning**.
- [ ] **W4.6 — Demo scenarios** (§18):
  - [ ] **Demo A, successful approved refund:**
    1. The customer submits a complaint.
    2. The agent retrieves records and policy.
    3. The reviewer sees the evidence and approves.
    4. The mock provider records the refund.
    5. The ticket receives the verified status.
  - [ ] **Demo B, correctly blocked request:**
    1. The customer requests an ineligible refund.
    2. The system explains the applicable policy.
    3. No payment action occurs.
    4. The case is escalated if appropriate.
  - [ ] **Demo C, failure recovery:**
    1. The provider accepts a refund but the response times out.
    2. The application restarts.
    3. The workflow reconciles the existing operation.
    4. No second refund is created.
- [ ] **W4.7 — README** (§19). It explains:
  1. the customer problem and supported scope
  2. architecture and workflow
  3. setup and seed-data instructions
  4. tool contracts and approval design
  5. refund idempotency and recovery behavior
  6. evaluation methodology and measured results
  7. known limitations
  8. demo link

  It also documents model configuration, embedding configuration, required environment variables and setup commands (§17).
- [ ] **W4.8 — Architecture diagram** (§16). Export from [`high-level-diagram.md`](high-level-diagram.md) into `docs/`.
- [ ] **W4.9 — Demo recording** (§16).
- [ ] **W4.10 — Résumé entry** (§19). Fill the placeholders **only after measuring**:
  > Built a customer support resolution system using Python, LangGraph, FastAPI, and PostgreSQL, integrating policy retrieval, customer APIs, human-approved refunds, and auditable ticket updates.
  > Achieved [X%] verified resolution on [N] held-out synthetic cases; tested duplicate-refund prevention and recovery across [M] retry, concurrency, and crash scenarios.

  Do not describe synthetic test results as production performance.

---

## Final checklist (PDF §20)

Confirm before sharing the project. Tasks that satisfy each item are in brackets.

- [ ] Customer access is derived from authenticated identity. [W1.5, W2.5]
- [ ] Order ownership is checked server-side. [W1.12]
- [ ] Policies have versions and citations. [W1.1, W2.1, W2.2]
- [ ] Model outputs use validated schemas. [W2.7, W2.15]
- [ ] Financial eligibility is enforced in code. [W1.14, W2.12]
- [ ] Refunds require approval bound to an exact proposal. [W3.1, W3.2, W3.7]
- [ ] Expired or changed proposals require fresh approval. [W3.6]
- [ ] Retries and concurrent requests cannot duplicate refunds. [W1.4, W3.9, W3.10]
- [ ] Unknown payment outcomes trigger reconciliation. [W3.9, W3.12]
- [ ] Ticket updates reflect verified external status. [W3.11, W3.14]
- [ ] Workflow state survives a restart. [W3.3, W3.13]
- [ ] Evaluation checks business state, not just response text. [W4.2]
- [ ] Results distinguish automated and human-assisted resolution. [W4.3]
- [ ] The README documents scope, tradeoffs, and limitations. [P0.5, W4.7]

---

## Open questions (decide before Week 1)

| # | Question | Why it matters |
|---|---|---|
| Q1 | Where does the **mock ticketing API** live? §5 lists it, but the §17 tree only has `mock_services/payments/` and `mock_services/shipping/`. Options: add `mock_services/ticketing/`, or treat the `tickets` table plus `update_ticket` as the ticketing system and inject faults there. | Needed for the *"refund succeeds, ticket update fails"* failure test. **Decided (2026-10-24): Option A, `mock_services/ticketing/`** (external helpdesk mock, port 8003). The app's `tickets` table stays the local record, and its status changes only after the ticketing API confirms. |
| Q2 | There is no `payments` table in the §7 schema. Confirm that payment and refund state come only from the mock payment API. | Affects `get_payment_status` and the refundable-balance calculation. |
| Q3 | Embeddings: pgvector inside PostgreSQL (optional per §5), or a simpler first pass? Which embedding model? | Affects W2.1, docker-compose, and `.env.example`. |
| Q4 | Which LLM provider and model, and what per-run step and cost limits? | Affects W2.9–W2.15 and the operational metrics. |
| Q5 | Customer and reviewer auth for the demo: static tokens, simple JWT, or an external IdP subject? | Affects W1.5, W1.6, and the authorization eval cases. |
| Q6 | Store currency (the PDF example uses `INR`) and the approval expiry duration. | One store and one currency (§3). Expiry drives W3.6. |
| Q7 | Prototype inline execution only, or the `202 Accepted` + worker model from the start? | Affects W3.12 and W3.13 and the crash-recovery demo. |
