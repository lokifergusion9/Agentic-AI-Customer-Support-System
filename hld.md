# High-Level Architecture — Customer Support Resolution Agent

Source: `Project.pdf`, *"A practical Agentic AI project for aspiring Forward Deployed Engineers"*.
Stack: **Python · LangGraph · FastAPI · PostgreSQL** (pgvector optional). All customer data is synthetic and every external service is mocked.

> **Core rule (PDF §4):** the model *proposes* actions. The application decides whether they are permitted and executes them through controlled tools.

Diagrams in this file:

1. [System architecture](#1-system-architecture)
2. [Resolution workflow (LangGraph)](#2-resolution-workflow-langgraph)
3. [Human approval: pause / resume](#3-human-approval--pause--resume)
4. [Refund execution, verification and reconciliation](#4-refund-execution-verification-and-reconciliation)
5. [Failure recovery](#5-failure-recovery)
6. [Database schema](#6-database-schema)
7. [API endpoints → components](#7-api-endpoints--components)
8. [LLM vs ordinary code boundary](#8-llm-vs-ordinary-code-boundary)
9. [Components → repository structure](#9-components--repository-structure)

---

## 1. System architecture

Four main components (PDF §5): **FastAPI app**, **LangGraph workflow**, **PostgreSQL**, and **mock external services**. RAG over support policies sits on top of PostgreSQL.

```mermaid
flowchart TB
    subgraph clients["Clients"]
        CUST["Customer demo UI<br/>complaint panel · timeline · final outcome"]
        REV["Reviewer UI<br/>approval queue · evidence · policy citations"]
        EVH["Eval harness<br/>evals/run_evals.py"]
    end

    subgraph api["FastAPI app — app/main.py · app/api · app/auth · app/schemas"]
        AUTHN["Authentication & authorization<br/>identity derived from the request"]
        ROUTES["Endpoints<br/>/tickets · /runs · /approvals · /health"]
        VALID["Input & response validation"]
        RUNNER["Run executor<br/>prototype: inline · robust: 202 Accepted + worker"]
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
```

Notes:

- **Identity comes from the authenticated request.** The model never picks which customer it acts for, and it never supplies reviewer identity, approval status or payment credentials (PDF §8).
- **Least privilege per agent.** The investigation agent has no refund-execution tool. Write tools are reachable only through the resolution proposal step and the code-owned execute step (PDF §6).
- **The checkpoint is not the financial ledger.** Refund accounting lives in application tables, with explicit constraints and transactions (PDF §7).
- **Mock services can return errors and delayed results**, so realistic failures can be tested (PDF §5).

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

Execution approach: the **prototype** runs short workflows inline and returns when they finish or pause. The **robust version** persists a job, returns `202 Accepted` with the run ID, executes on a worker (FastAPI `BackgroundTasks` or Celery), persists progress, and lets clients poll status.

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
│   └── shipping/          # shipment status + fault injection
├── migrations/
├── scripts/
│   ├── seed_data.py
│   └── ingest_policies.py
├── evals/
│   ├── datasets/
│   ├── run_evals.py
│   └── reports/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── failure_scenarios/
├── docs/
├── docker-compose.yml
├── .env.example
├── pyproject.toml
└── README.md
```

The PDF lists a **ticketing API** among the mock services (§5), but its repository tree only has `payments/` and `shipping/` under `mock_services/`. See the open question in [`tasks.md`](tasks.md).
