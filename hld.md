# High-Level Architecture — Customer Support Resolution Agent

Sources:

- `Project.pdf`, *"A practical Agentic AI project for aspiring Forward Deployed Engineers"*: the resolution agent (§1–§9 of this file, cited as `PDF §n`).
- `Project_2.pdf`, *"A complete project guide for engineering students and aspiring Forward Deployed Engineers"*: the evaluation platform (§10, cited as `E§n`). Merged on 2026-10-27 in a scoped form; the resolution agent is the system it evaluates.

Stack: **Python · LangGraph · FastAPI · PostgreSQL** (pgvector optional). All customer data is synthetic and every external service is mocked.

> **Core rule (PDF §4):** the model *proposes* actions. The application decides whether they are permitted and executes them through controlled tools.
>
> **Evaluation rule (E§1):** an agent's final message is evidence to inspect, not proof that the task succeeded.

Diagrams in this file:

- [High-Level Architecture — Customer Support Resolution Agent](#high-level-architecture--customer-support-resolution-agent)
  - [1. System architecture](#1-system-architecture)
  - [2. Resolution workflow (LangGraph)](#2-resolution-workflow-langgraph)
  - [3. Human approval — pause / resume](#3-human-approval--pause--resume)
  - [4. Refund execution, verification and reconciliation](#4-refund-execution-verification-and-reconciliation)
  - [5. Failure recovery](#5-failure-recovery)
    - [5a. Worker crash / application restart](#5a-worker-crash--application-restart)
    - [5b. Failure → expected behavior (PDF §14)](#5b-failure--expected-behavior-pdf-14)
  - [6. Database schema](#6-database-schema)
  - [7. API endpoints → components](#7-api-endpoints--components)
  - [8. LLM vs ordinary code boundary](#8-llm-vs-ordinary-code-boundary)
  - [9. Components → repository structure](#9-components--repository-structure)
  - [10. Evaluation platform](#10-evaluation-platform)
    - [10a. Components](#10a-components)
    - [10b. One evaluation attempt](#10b-one-evaluation-attempt)
    - [10c. Agent-visible vs evaluator-visible data](#10c-agent-visible-vs-evaluator-visible-data)
    - [10d. `eval` schema](#10d-eval-schema)
    - [10e. Execution and evaluation states](#10e-execution-and-evaluation-states)

---

## 1. System architecture

Four main components (PDF §5): **FastAPI app**, **LangGraph workflow**, **PostgreSQL**, and **mock external services**. RAG over support policies sits on top of PostgreSQL.

```mermaid
flowchart TB
    subgraph clients["Clients"]
        CUST["Customer demo UI<br/>complaint panel · timeline · final outcome"]
        REV["Reviewer UI<br/>approval queue · evidence · policy citations"]
        EVH["Evaluation platform — evals/<br/>adapters · reviewer simulator · evaluators<br/>details in §10"]
    end

    subgraph api["FastAPI app — app/main.py · app/api · app/auth · app/schemas"]
        AUTHN["Authentication & authorization<br/>identity derived from the request"]
        ROUTES["Endpoints<br/>/tickets · /runs · /approvals · /health"]
        VALID["Input & response validation"]
        RUNNER["Run executor<br/>prototype: inline · robust: 202 Accepted + worker<br/>Postgres jobs queue, SKIP LOCKED (Q7)"]
    end

    subgraph wf["LangGraph workflow — app/workflows · app/agents"]
        CLS["Classify complaint"]
        INV["Investigation agent<br/>read-only fact tools"]
        POLA["Policy agent<br/>search_policy only"]
        RESA["Resolution agent<br/>structured proposal"]
        RULES["Business-rule validation<br/>ordinary code"]
        GATE["Approval gate<br/>interrupt()"]
        EXEC["Execute → verify → update ticket<br/>ordinary code"]
    end

    subgraph tools["Typed tools & services — app/tools · app/services"]
        RT["Read tools<br/>get_order · get_return_status · get_payment_status<br/>get_refund_status · search_policy · get_ticket"]
        WT["Write tools<br/>create_refund_proposal · execute_approved_refund<br/>update_ticket"]
        ELIG["Eligibility · amounts · refundable balance<br/>ownership checks · deterministic"]
    end

    subgraph rag["RAG — app/policies · scripts/ingest_policies.py"]
        PDOCS["Synthetic policy documents<br/>id · version · effective dates"]
        EMB["Section chunking + embeddings"]
        RETR["Version-filtered retrieval<br/>citations · missing / conflicting evidence"]
    end

    subgraph pg["PostgreSQL — app/db · migrations/"]
        APPDB[("Application tables<br/>customers · orders · tickets · runs<br/>proposals · approvals · refund_operations · audit_events")]
        VEC[("policy_documents · policy_chunks<br/>pgvector optional")]
        CKPT[("LangGraph checkpoint tables<br/>logically separate from app tables")]
        EVALDB[("eval schema<br/>experiments · runs · attempts · scores<br/>ground truth: no agent access")]
    end

    subgraph mocks["Mock external services — mock_services/"]
        PAY["Payment / refund API<br/>idempotency-key contract"]
        SHIP["Shipping API"]
        TIX["Ticketing API"]
    end

    LLM["LLM provider"]

    CUST --> AUTHN
    REV --> AUTHN
    EVH --> AUTHN
    AUTHN --> ROUTES --> VALID --> RUNNER --> CLS
    CLS --> INV & POLA
    INV & POLA --> RESA --> RULES --> GATE --> EXEC

    CLS -.-> LLM
    INV -.-> LLM
    POLA -.-> LLM
    RESA -.-> LLM

    INV --> RT
    POLA --> RETR
    RESA --> WT
    RULES --> ELIG
    EXEC --> WT
    EXEC --> ELIG

    PDOCS --> EMB --> VEC
    RETR --> VEC
    RT --> APPDB
    RT --> SHIP
    RT --> PAY
    WT --> APPDB
    WT --> PAY
    WT --> TIX
    ELIG --> APPDB

    RUNNER --> CKPT
    GATE --> CKPT
    ROUTES --> APPDB
    EVH --> EVALDB
    EVH -.->|"reset + load fixture per attempt"| APPDB
    EVH -.->|"reset + faults"| mocks
```

Notes:

- **Identity comes from the authenticated request.** The model never picks which customer it acts for, and it never supplies reviewer identity, approval status or payment credentials (PDF §8).
- **Least privilege per agent.** The investigation agent has no refund-execution tool. Write tools are reachable only through the resolution proposal step and the code-owned execute step (PDF §6).
- **The checkpoint is not the financial ledger.** Refund accounting lives in application tables, with explicit constraints and transactions (PDF §7).
- **Mock services can return errors and delayed results**, so realistic failures can be tested (PDF §5).
- **The evaluation platform drives the agent like a client.** It creates tickets and starts runs through the same API and tools, so every safety check applies to it too. Only the platform reads the `eval` schema (§10).

---

## 2. Resolution workflow (LangGraph)

This is the PDF §4 foundation flow, with branches for the three supported complaint types (order status, refund request, damaged item). Nodes marked **[LLM]** use the model. Everything else is ordinary code.

```mermaid
flowchart TD
    A["Customer submits complaint<br/>POST /tickets → POST /tickets/{ticket_id}/runs"] --> B["Authenticate & validate request"]
    B -->|"order belongs to another customer"| DENY["Deny access<br/>reveal no order details"]
    B --> C["Classify complaint [LLM]<br/>order_status · refund_request · damaged_item"]
    C -->|"unsupported / out of scope"| ESC
    C --> D["Retrieve customer & order facts + relevant policies<br/>Investigation agent ∥ Policy agent [LLM]"]
    D --> E{"Missing information?"}
    E -->|"yes"| Q["Ask a targeted clarification question<br/>run waits for POST /runs/{run_id}/messages"]
    Q --> D
    E -->|"no"| F["Create proposed resolution [LLM]<br/>validated structured proposal"]
    F --> G{"Validate against business rules<br/>ownership · eligibility · amount · policy"}

    G -->|"answer only<br/>e.g. order status, ineligible refund"| UPD
    G -->|"refund proposed"| H["Persist immutable proposal<br/>create pending approval"]
    G -->|"no policy · conflicting evidence<br/>needs human judgement"| ESC["Escalate to human"]

    H --> I["Pause workflow — interrupt()"]
    I --> J{"Approve / reject / expire"}
    J -->|"reject"| RJ["Record decision<br/>follow rejection path"]
    J -->|"expire"| H
    J -->|"approve"| K["Revalidate approved action<br/>eligibility · refundable balance · payload hash"]
    K -->|"order / amount / material details changed"| H
    K -->|"still valid"| L["Execute tool call<br/>execute_approved_refund"]
    L --> M["Verify external result"]
    M -->|"uncertain"| REC["Reconcile by idempotency key"]
    REC --> M
    M -->|"confirmed"| UPD["Update ticket and audit trail"]
    RJ --> UPD
    ESC --> UPD
    UPD --> Z["Return accurate customer update<br/>claims no more than the API confirmed"]

    classDef llm fill:#ede7f6,stroke:#5e35b1,color:#1a1a1a
    classDef gate fill:#fff3e0,stroke:#ef6c00,color:#1a1a1a
    classDef money fill:#e8f5e9,stroke:#2e7d32,color:#1a1a1a
    classDef stop fill:#ffebee,stroke:#c62828,color:#1a1a1a
    class C,D,F llm
    class G,I,J,K gate
    class L,M,REC money
    class DENY,ESC stop
```

Per complaint type (PDF §3):

| Complaint | Typical path |
|---|---|
| **A. Order status** | facts (order + shipment) → answer only → update ticket |
| **B. Refund request** | facts + refund policy → proposal → approval → execute → verify → update ticket |
| **C. Damaged item** | facts + damaged-product policy → collect required info → resolution proposal **or** escalate |

First-version limits: one store, one currency, **human approval for every refund**, one customer issue per workflow, and no chargebacks, fraud adjudication or complex partial returns.

---

## 3. Human approval — pause / resume

PDF §11. The approval is tied to one **exact, immutable proposal**. An approval message in chat does not authorize anything: the execution service checks the stored approval and the authenticated reviewer's decision.

```mermaid
sequenceDiagram
    autonumber
    participant WF as LangGraph run (thread_id)
    participant DB as PostgreSQL
    participant API as FastAPI
    actor RV as Reviewer

    WF->>DB: INSERT action_proposals (payload, payload_hash) — immutable
    WF->>DB: INSERT approvals (proposal_id, expires_at, decision = pending)
    WF->>DB: checkpoint saved (checkpoint tables)
    WF-->>API: interrupt() — run status = awaiting_approval

    RV->>API: GET /approvals
    API->>API: authenticate reviewer + validate reviewer permissions
    API-->>RV: customer, order, complaint summary, proposed action + amount,<br/>evidence, policy citations, expiry time
    RV->>API: POST /approvals/{approval_id}/decision
    API->>DB: save decision, reviewer_id (from auth), decided_at
    API->>WF: Command(resume=...) on the same thread_id

    Note over WF: The interrupted node restarts on resume.<br/>No financial side effects before interrupt().
    WF->>DB: re-check expiry, payload_hash, eligibility, refundable balance
    alt rejected
        WF->>DB: record rejection, follow rejection path
    else expired or material details changed
        WF->>DB: invalidate old approval, request a new one
    else approved and still valid
        WF->>WF: continue to the separate execute node
    end
```

---

## 4. Refund execution, verification and reconciliation

PDF §12. The hard test case: *the payment service accepts the refund, but the app times out before it gets the response.* Retrying blindly with a new request would create a duplicate refund.

```mermaid
sequenceDiagram
    autonumber
    participant EX as Execute node
    participant DB as PostgreSQL
    participant PAY as Mock payment API

    EX->>DB: BEGIN — SELECT order / refund rows FOR UPDATE
    EX->>DB: validate approval, check refundable balance,<br/>INSERT refund_operations (stable idempotency_key) and reserve amount
    EX->>DB: COMMIT — no transaction held during the network call
    EX->>PAY: refund request (idempotency_key, amount_minor)

    alt provider responds with success
        PAY-->>EX: provider_reference, status
        EX->>DB: persist response, mark operation succeeded
    else timeout or unclear result
        EX->>DB: mark operation unknown / pending_reconciliation
        EX->>PAY: query by idempotency_key or provider_reference
        alt provider already has the operation
            PAY-->>EX: existing operation
            EX->>DB: mark succeeded — no second refund created
        else provider has no record and contract allows retry
            EX->>PAY: retry with the SAME key and SAME payload
        end
    else provider rejects
        PAY-->>EX: structured error (error_code, retryable, message)
        EX->>DB: mark failed, release reservation
    end

    EX->>EX: verify external result before any customer-facing claim
    Note over PAY: Mock contract: same key + same payload returns the same operation.<br/>Same key + changed payload is rejected.
```

Ledger rules (PDF §7):

- Money is stored as **integer minor units** (paise / cents).
- `refund_operations.idempotency_key` has a **unique constraint**.
- Successful refunds **plus** amounts reserved by pending operations can never exceed the refundable balance.
- Concurrent requests are serialized with `SELECT ... FOR UPDATE` or equivalent.
- A checkpoint alone does not guarantee that an external action happens exactly once.

---

## 5. Failure recovery

### 5a. Worker crash / application restart

```mermaid
flowchart TD
    S["Application / worker starts"] --> R["Find unfinished workflow_runs"]
    R --> L["Load LangGraph checkpoint by thread_id"]
    L --> O{"Open refund_operation for this run?"}
    O -->|"no"| RES["Resume workflow from checkpoint"]
    O -->|"reserved · unknown · pending_reconciliation"| QP["Query payment provider by idempotency_key"]
    QP -->|"provider confirms refund"| MS["Mark succeeded<br/>do NOT call refund again"]
    QP -->|"provider has no record"| RT{"Retry allowed by contract?"}
    RT -->|"yes"| RK["Retry with the same key + payload"]
    RT -->|"no"| ESC["Escalate for manual review"]
    MS --> TK["Retry only the pending ticket update"]
    RK --> TK
    TK --> RES
    RES --> DONE["Verified outcome + audit event"]
```

### 5b. Failure → expected behavior (PDF §14)

```mermaid
flowchart LR
    subgraph input["Input & access"]
        F1["Missing order information"] --> B1["Ask targeted clarification question"]
        F2["Order belongs to another customer"] --> B2["Deny access without revealing details"]
    end
    subgraph policy["Policy"]
        F3["No applicable policy found"] --> B3["Escalate — never invent eligibility"]
        F4["Conflicting policy evidence"] --> B4["Flag the conflict for review"]
    end
    subgraph model["Model & tools"]
        F5["Temporary read API failure"] --> B5["Retry with bounded backoff"]
        F6["Invalid model output"] --> B6["Validate, limited repair, then stop or escalate"]
        F12["Excessive workflow loops"] --> B12["Stop at a configured step or cost limit"]
    end
    subgraph approval["Approval"]
        F7["Approval rejected"] --> B7["Record decision, follow rejection path"]
        F8["Approval expired"] --> B8["Require a new valid approval"]
    end
    subgraph money["Execution"]
        F9["Refund outcome uncertain"] --> B9["Reconcile before retrying or claiming success"]
        F10["Refund succeeds, ticket update fails"] --> B10["Retry only the ticket update"]
        F11["Worker crashes"] --> B11["Resume from persisted state, reconcile side effects"]
    end
```

---

## 6. Database schema

The tables and "important fields" are exactly the PDF §7 suggested schema. Column types are a suggested mapping only. Application tables are kept **logically separate** from LangGraph checkpoint tables.

```mermaid
erDiagram
    customers ||--o{ orders : places
    customers ||--o{ tickets : opens
    orders ||--|{ order_items : contains
    orders ||--o{ returns : has
    orders |o--o{ tickets : "is about"
    orders ||--o{ refund_operations : "refunded by"
    tickets ||--o{ workflow_runs : "resolved by"
    workflow_runs ||--o{ action_proposals : produces
    workflow_runs ||--o{ audit_events : logs
    action_proposals ||--o{ approvals : "decided by"
    action_proposals ||--o{ refund_operations : "executed as"
    policy_documents ||--|{ policy_chunks : "split into"

    customers {
        uuid id PK
        text external_auth_subject UK
        text name
        text email
    }
    orders {
        text id PK
        uuid customer_id FK
        text status
        text currency
        bigint paid_amount_minor
        timestamptz delivered_at
    }
    order_items {
        uuid id PK
        text order_id FK
        text product_id
        int quantity
        bigint paid_amount_minor
    }
    returns {
        text id PK
        text order_id FK
        text status
        timestamptz received_at
        text inspection_status
    }
    tickets {
        uuid id PK
        uuid customer_id FK
        text order_id FK
        text category
        text status
        text summary
    }
    workflow_runs {
        uuid id PK
        uuid ticket_id FK
        text thread_id "LangGraph thread"
        text status
        timestamptz started_at
        timestamptz completed_at
    }
    policy_documents {
        text id PK
        text version
        date effective_from
        date effective_to
        text source_uri
    }
    policy_chunks {
        uuid id PK
        text document_id FK
        text content
        vector embedding "pgvector optional"
        jsonb metadata
    }
    action_proposals {
        uuid id PK
        uuid run_id FK
        text action_type
        jsonb payload
        text payload_hash
        text status
    }
    approvals {
        uuid id PK
        uuid proposal_id FK
        text reviewer_id
        text decision
        timestamptz expires_at
        timestamptz decided_at
    }
    refund_operations {
        uuid id PK
        text order_id FK
        uuid proposal_id FK
        bigint amount_minor
        text idempotency_key UK
        text provider_reference
        text status
    }
    audit_events {
        uuid id PK
        uuid run_id FK
        text actor
        text event_type
        jsonb redacted_payload
        timestamptz created_at
    }
```

There is no `payments` table in the PDF schema. Payment and refund status are read from the **mock payment API** (`get_payment_status`, `get_refund_status`).

Additions beyond the PDF §7 schema (all in the `support` schema, added by later migrations):

| Addition | Task | Purpose |
|---|---|---|
| `jobs` table | W3.13 (Q7) | Postgres job queue: `job_type` (`workflow_run` · `eval_attempt`), `payload`, `status`, `available_at`, `attempts`, `worker_id`, `lease_token`, `lease_expires_at` |
| `audit_events.sequence_number` | W3.15 | Per-run event order, unique per run, so the evaluation platform can replay a trace exactly |
| `model_call` · `tool_requested` · `tool_rejected` · `tool_finished` · `agent_finished` event types | W2.16, W3.15 | Token usage, latency, blocked attempts vs executed effects, stop reason |

The evaluation platform's own tables live in a separate `eval` schema (§10d).

---

## 7. API endpoints → components

Suggested endpoints, exactly as in PDF §13. Authorization is checked on **every** endpoint.

| Endpoint | Purpose | Caller | Touches |
|---|---|---|---|
| `POST /tickets` | Create a customer support ticket | Customer | `tickets` |
| `POST /tickets/{ticket_id}/runs` | Start a resolution workflow | Customer / system | `workflow_runs`, LangGraph |
| `GET /runs/{run_id}` | Retrieve status and output | Customer / reviewer | `workflow_runs`, checkpoint |
| `GET /runs/{run_id}/events` | Retrieve a redacted event timeline | Customer / reviewer | `audit_events` |
| `GET /approvals` | List approvals available to the reviewer | Reviewer | `approvals`, `action_proposals` |
| `POST /approvals/{approval_id}/decision` | Approve or reject a proposal | Reviewer | `approvals` → `Command(resume=...)` |
| `POST /runs/{run_id}/messages` | Supply additional customer information | Customer | resumes the clarification wait |
| `GET /health` | Check application health | Ops | DB + mocks |

Execution approach: the **prototype** runs short workflows inline and returns when they finish or pause. The **robust version** persists a job, returns `202 Accepted` with the run ID, executes on a separate worker process that claims jobs from the Postgres `jobs` table with `SELECT ... FOR UPDATE SKIP LOCKED` (Q7, W3.13), persists progress, and lets clients poll status.

Evaluation endpoints (E§24, W5.4). All sit under `/eval` so they never collide with the agent's `/runs` endpoints, and only the `operator` role may call them (Q12):

| Endpoint | Purpose | Touches |
|---|---|---|
| `POST /eval/datasets/import` · `GET /eval/datasets` | Validate and import a dataset version; list them | `eval.datasets`, `dataset_versions`, `task_versions` |
| `POST /eval/agent-configs` · `GET /eval/agent-configs` | Register an immutable configuration; list them | `eval.agent_configs`, `prompt_versions` |
| `POST /eval/experiments` | Create experiment, runs and jobs in one transaction (`Idempotency-Key`) | `eval.experiments`, `eval.runs`, `support.jobs` |
| `GET /eval/experiments/{id}` | Progress and scorecard | `eval.runs`, `eval.scores` |
| `GET /eval/experiments/{id}/runs` | Paginated runs with filters | `eval.runs` |
| `POST /eval/experiments/{id}/cancel` | Request cancellation | `eval.experiments`, `support.jobs` |
| `GET /eval/runs/{id}` | Outcome, attempts, scores | `eval.runs`, `attempts`, `scores` |
| `GET /eval/attempts/{id}/events` | Ordered trace | `eval.trace_events` |
| `POST /eval/comparisons` | Compare compatible experiments | `eval.*` |
| `GET /eval/experiments/{id}/export` | Download the JSON / CSV report | `eval.artifacts` |

---

## 8. LLM vs ordinary code boundary

```mermaid
flowchart LR
    subgraph LLMside["LLM — proposes"]
        L1["Complaint classification"]
        L2["Investigation: evidence summary"]
        L3["Policy: citations + interpretation"]
        L4["Resolution: structured proposal"]
        L5["Customer-facing wording"]
    end
    subgraph Codeside["Ordinary code — decides & executes"]
        C1["Authentication"]
        C2["Access control / ownership"]
        C3["Amount calculations"]
        C4["Eligibility enforcement"]
        C5["Approval validation"]
        C6["API execution"]
        C7["Retry limits"]
        C8["State transitions"]
    end
    L4 -->|"schema-validated proposal"| C4
    C6 -->|"verified result only"| L5
```

| Agent | Responsibility | Output | Tools allowed |
|---|---|---|---|
| Investigation agent | Gather customer, order, return and payment facts | Structured evidence summary | read tools only — **no refund execution** |
| Policy agent | Retrieve applicable policy passages | Citations and policy interpretation | `search_policy` |
| Resolution agent | Combine facts and policy into a proposed action | Structured resolution proposal | `create_refund_proposal` (proposal only, moves no money) |

Investigation and policy retrieval run **concurrently** when they are independent. If policy selection depends on the product or order date, facts are gathered first. A **single-agent baseline** is built first and compared against this specialist design (PDF §6).

---

## 9. Components → repository structure

Repository layout as in PDF §17:

```text
customer-support-resolution-agent/
├── app/
│   ├── main.py            # FastAPI entrypoint
│   ├── api/               # endpoints (§7 table above)
│   ├── auth/              # customer + reviewer authentication / authorization
│   ├── schemas/           # validated I/O + specialist output schemas
│   ├── agents/            # investigation · policy · resolution
│   ├── workflows/         # LangGraph graph, state, routing, interrupt/resume
│   ├── tools/             # typed read / write tools
│   ├── services/          # eligibility, refunds, reconciliation, tickets, audit
│   ├── policies/          # policy retrieval (RAG)
│   └── db/                # models, sessions, checkpointer config
├── mock_services/
│   ├── payments/          # refund API with idempotency contract + fault injection
│   ├── shipping/          # shipment status + fault injection
│   └── ticketing/         # ticket status + notes updates + fault injection (Q1)
├── migrations/
├── scripts/
│   ├── seed_data.py
│   └── ingest_policies.py
├── evals/                 # evaluation platform (§10); app/ never imports from here
│   ├── datasets/
│   │   └── support-v1/
│   │       ├── tasks/     # one YAML per task: agent view + evaluator view
│   │       └── splits/    # dev · validation · test · pilot manifests
│   ├── adapters/          # base.py (AgentAdapter) · scripted.py · langgraph.py
│   ├── evaluators/        # outcome · policy · trajectory · response
│   ├── environment.py     # per-attempt reset, fixture loading, snapshots
│   ├── reviewer.py        # deterministic reviewer simulator
│   ├── run_evals.py       # create an experiment from a manifest
│   └── reports/           # exported JSON / CSV per experiment
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── failure_scenarios/
│   └── evals/             # evaluator, isolation, worker and accounting tests
├── docs/
├── Makefile               # setup · migrate · seed · test · smoke · evaluate · report
├── docker-compose.yml     # + worker service (W3.13)
├── .env.example
├── pyproject.toml
└── README.md
```

The PDF lists a **ticketing API** among the mock services (§5), but its repository tree only has `payments/` and `shipping/` under `mock_services/`. **Decided (Q1 in [`tasks.md`](tasks.md), 2026-10-24):** the ticketing API is a separate mock service in `mock_services/ticketing/` (port 8003). The app's `tickets` table stays the local record, and its status changes only after the ticketing API confirms.

The `evals/` subfolders, `tests/evals/` and the `Makefile` come from `Project_2.pdf` (E§26, E§30), adapted to this repository instead of its separate `agent-evaluation-platform/` layout.

---

## 10. Evaluation platform

`Project_2.pdf`, scoped (Weeks 4–5 in [`tasks.md`](tasks.md)). It answers one question (E§1): *"If I change the agent's model, prompt, tools or orchestration, does it become more reliable, and at what cost?"* The resolution agent from §1–§9 is the system under test.

Objects, used consistently (E§4):

| Object | Meaning |
|---|---|
| Task | One problem with initial conditions and success criteria (a YAML file) |
| Dataset version | An immutable collection of task versions (`support-v1`) |
| Agent configuration | A fixed combination of model, prompt, architecture and limits |
| Experiment | A planned evaluation of configurations against a dataset split |
| Run | One configuration × one task × one repetition |
| Attempt | A concrete execution of a run; an infrastructure retry is a new attempt |
| Trace | The ordered events of an attempt: model calls, tool requests and results, decisions |
| Evaluator | Deterministic code that scores one property |
| Score | An evaluator's result, with evidence and applicability |
| Artifact | A final-state snapshot or an exported report |

Scope (decided 2026-10-27): one custom dataset of 50 tasks, deterministic evaluators only, simulated approvals, one attempt at a time. Public benchmarks, an LLM judge, simulated multi-turn users and concurrent attempts are extensions.

### 10a. Components

```mermaid
flowchart TB
    subgraph clients["Clients"]
        CLI["CLI — evals/run_evals.py · Makefile"]
        DASH["Dashboard — experiments · run detail · comparison"]
    end

    subgraph evapi["FastAPI — /eval/* (operator role, Q12)"]
        EAPI["Datasets · agent configs · experiments<br/>comparisons · export"]
    end

    subgraph worker["Worker process (shared with W3.13)"]
        CLAIM["Claim eval_attempt job<br/>SKIP LOCKED + lease token"]
        ENV["Environment manager — evals/environment.py<br/>advisory lock · reset · load fixture · snapshot"]
        ADP["Agent adapter<br/>scripted · LangGraph (single / specialist)"]
        SIM["Reviewer simulator<br/>approve / reject / expire"]
        EVALS["Evaluators<br/>outcome · policy · trajectory · response"]
    end

    subgraph agent["System under test (§1–§9)"]
        AAPI["Agent API + typed tools<br/>ownership · approval · idempotency enforced"]
    end

    subgraph pg["PostgreSQL"]
        SUP[("support schema<br/>business tables · audit_events · jobs")]
        EV[("eval schema<br/>tasks · configs · experiments · runs<br/>attempts · trace_events · artifacts · scores")]
    end

    MOCKS["Mock payment · shipping · ticketing<br/>faults keyed by logical operation"]
    LLM["Gemini Flash / Flash-Lite (Q9)"]

    CLI --> EAPI
    DASH --> EAPI
    EAPI --> EV
    EAPI -->|"one transaction: experiment + runs + jobs"| SUP
    CLAIM --> SUP
    CLAIM --> ENV
    ENV -->|"reset + fixture"| SUP
    ENV -->|"reset + fixture + faults"| MOCKS
    ENV --> ADP
    ADP --> AAPI
    AAPI --> SUP
    AAPI --> MOCKS
    AAPI -.-> LLM
    SIM -->|"decision via the real approval service"| AAPI
    ENV -->|"snapshot + copy trace"| EV
    EVALS --> EV

    classDef hidden fill:#fff3e0,stroke:#ef6c00,color:#1a1a1a
    class EV,EVALS,SIM hidden
```

Notes:

- **The agent is driven like any client.** The LangGraph adapter creates a ticket under the task customer's JWT and starts a run. The scripted adapter calls the same typed tools. So the platform tests the real enforcement, not a shortcut around it.
- **One worker, two job types.** `workflow_run` (W3.13) and `eval_attempt` (W4.8) share the queue, leases and recovery.
- **Orange components see hidden data.** Nothing on the agent side reads the `eval` schema.

### 10b. One evaluation attempt

```mermaid
sequenceDiagram
    autonumber
    participant W as Worker
    participant EV as eval schema
    participant ENV as Environment manager
    participant SUP as support schema + mocks
    participant AD as Agent adapter
    participant AG as Agent (API + tools)
    participant SIM as Reviewer simulator
    participant E as Evaluators

    W->>SUP: claim eval_attempt job (SKIP LOCKED, lease token) — commit
    W->>EV: INSERT attempt (attempt_number, environment_id)
    W->>ENV: prepare(task)
    ENV->>SUP: advisory lock · reset business tables + mocks
    ENV->>SUP: load fixture · set clock to as_of · register faults
    W->>AD: run(agent view only, limits)
    AD->>AG: create ticket + start run (customer JWT)
    AG->>SUP: tool calls, model calls, audit events
    opt run waits for approval
        AD->>SIM: approval pending
        SIM->>AG: approve / reject / expire via approval service
        AG->>SUP: resume, execute or follow rejection path
    end
    AG-->>AD: final response + stop reason
    ENV->>EV: snapshot final state (DB rows + provider records) into artifacts
    ENV->>EV: copy audit events into trace_events
    W->>E: evaluate(evaluator view, snapshot, trace)
    E->>EV: INSERT scores (one per dimension)
    W->>EV: execution_status + evaluation_status (requires current lease token)
    W->>SUP: finish job — release advisory lock
```

If the worker dies mid-attempt, the lease expires, the sweep requeues the job, and a **new attempt with a fresh environment** runs. The stale worker cannot finalize because its lease token is no longer current. Agent behavior failures are scored and never retried (E§17).

### 10c. Agent-visible vs evaluator-visible data

E§5, E§8: the agent and the evaluators share one codebase but have different access paths and data views.

```mermaid
flowchart LR
    TASK["Task YAML"] --> AV & XV
    subgraph agentside["Agent-visible"]
        AV["agent_input<br/>user message · customer subject · optional order ID"]
        TOOLS["Facts learned only through tools<br/>(ownership-checked)"]
    end
    subgraph evalside["Evaluator-visible only"]
        XV["environment_fixture · faults<br/>reviewer_behavior<br/>expected outcome · invariants<br/>required behavior · response checks"]
    end
    AV --> AGENT["Agent"]
    TOOLS --> AGENT
    XV --> FIX["Fixture loaded into DB + mocks"]
    FIX --> TOOLS
    XV --> EVL["Evaluators"]
    AGENT -->|"trace + final state"| EVL
```

The worker loads the full fixture, but the agent receives only `agent_input`; it learns order facts by calling tools. A test proves the agent view never contains fixture, reviewer or evaluation keys (W4.1).

Evaluators (E§14):

| Evaluator | Reads | Checks |
|---|---|---|
| Outcome | Final-state snapshot | Correct order and amount, exactly one financial effect, ticket consistent, approval requested when required, or the expected non-action |
| Policy | Trace | No cross-customer access, no refund without approval, no over-refund, no forbidden effect; **unsafe attempts and executed violations counted separately** |
| Trajectory | Trace | Tool choice, schema validity, arguments, partial-order preconditions, redundancy, handling of injected faults |
| Response | Final message + snapshot | The message claims nothing the state does not confirm |

**Verified success** = required outcome **and** no hard invariant violated **and** required response checks pass (E§20).

### 10d. `eval` schema

```mermaid
erDiagram
    datasets ||--|{ dataset_versions : "versioned as"
    dataset_versions ||--|{ task_versions : contains
    prompt_versions ||--o{ agent_configs : "used by"
    dataset_versions ||--o{ experiments : "evaluated in"
    experiments ||--|{ runs : plans
    agent_configs ||--o{ runs : "runs as"
    task_versions ||--o{ runs : "runs on"
    runs ||--|{ attempts : "executed as"
    attempts ||--o{ trace_events : records
    attempts ||--o{ artifacts : snapshots
    attempts ||--o{ scores : "scored by"
    price_versions ||--o{ agent_configs : "prices"

    dataset_versions {
        text id PK
        text dataset_id FK
        jsonb manifest "splits"
        text content_hash
    }
    task_versions {
        text id PK
        text task_id
        int version
        text category
        jsonb agent_view
        jsonb evaluator_view
        text content_hash
    }
    agent_configs {
        text id PK "immutable"
        text adapter
        text provider
        text model_id
        text prompt_version_id FK
        text architecture "single | specialist"
        jsonb limits
        text tool_schema_hash
    }
    experiments {
        text id PK
        text dataset_version_id FK
        text split
        jsonb manifest "immutable"
        text status
        text idempotency_key UK
    }
    runs {
        text id PK
        text experiment_id FK
        text agent_config_id FK
        text task_version_id FK
        int repetition
        text execution_status
        text evaluation_status
    }
    attempts {
        text id PK
        text run_id FK
        int attempt_number
        text environment_id
        text workflow_run_id
        text stop_reason
        text final_response
    }
    trace_events {
        text id PK
        text attempt_id FK
        int sequence_number
        text event_type
        jsonb redacted_payload
    }
    scores {
        text id PK
        text attempt_id FK
        text evaluator_name
        text evaluator_version
        text dimension
        bool applicable
        text status
        text reason_code
        jsonb evidence
    }
```

Constraints (E§16): `UNIQUE(experiment_id, agent_config_id, task_version_id, repetition)`, `UNIQUE(run_id, attempt_number)`, `UNIQUE(attempt_id, sequence_number)`, `UNIQUE(attempt_id, evaluator_name, evaluator_version, dimension)`. Traces and snapshots are **copied** into `eval` because the next attempt resets the `support` business tables (Q8, Q11).

### 10e. Execution and evaluation states

E§18: execution status and evaluation outcome are tracked separately.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> queued
    queued --> running : worker claims
    running --> completed
    running --> timed_out
    running --> budget_exceeded
    running --> infrastructure_error
    queued --> cancelled
    running --> cancelled
    infrastructure_error --> queued : bounded retry, new attempt + fresh environment
```

| Execution status | Evaluation status | Example |
|---|---|---|
| `completed` | `pass` | Correct approved refund, or a correct refusal |
| `completed` | `fail` | Confident final message, wrong result |
| `completed` | `fail` (stop reason `max_steps`) | Hit the turn limit |
| `infrastructure_error` | `pending` | Model provider unreachable after retries |
| any | `evaluation_error` | An evaluator raised; never a silent pass or agent failure |
| any | `not_applicable` | Per dimension, e.g. trajectory scores for a task needing no tools |

Metrics are computed from these states with explicit denominators (verified success, strict end-to-end success, coverage, unsafe attempt rate, executed violation rate, tool precision and recall, latency, tokens and estimated cost). Full definitions: W5.1 in [`tasks.md`](tasks.md).
