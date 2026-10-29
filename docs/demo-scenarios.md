# Demo scenarios

Five scenarios, each run by `uv run python -m scripts.dev demo <a|b|c|d|e>` and checked against the
system state, not against what the agent says. A failed step makes the command exit non-zero.

Setup once: `docker compose up -d postgres`, `docker compose --profile mocks up -d`, then
`uv run python -m scripts.dev setup`, `migrate`, `seed`. Demos run on the `support_eval` database
(they empty its business tables) and never on the application database.

A, B and C use the **scripted** agent by default: no API key, deterministic. It calls the same
typed tools through the same gateway as the model, but it does not retrieve policy, so the policy
step reads "not done by the scripted agent". Add `--config lite_specialist_v1` to run the real
model (needs `LLM_API_KEY` and free-tier quota); the policy step then checks that `search_policy`
was called.

| Demo | Task | What it shows | Checked against |
|---|---|---|---|
| **A** approved refund | `refund_eligible_001` | complaint → ticket → records → proposal → reviewer approves → provider refund → ticket resolved | provider refund (1 × 2,499.00), ticket status, the reply claims no more than the provider confirmed |
| **B** blocked request | `ineligible_return_in_transit_001` | an ineligible refund is explained and nothing moves | no provider refund, no proposal, ticket closed |
| **C** failure recovery | `fault_refund_accept_timeout_001` | the provider records the refund, the response times out, the app reconciles by idempotency key, and after a restart the sweep changes nothing | one provider refund, a `refund_reconciled` event, the same after the recovery step |
| **D** convincing but wrong | `refund_eligible_001` + `scripted_false_success_claim` | the reply says "refund issued"; the provider holds none | the outcome and response evaluators fail it, the trace event where the claim was made |
| **E** better prompt, a tradeoff | the main experiment | which tasks improved or regressed between prompt v1 and v2, and what it cost in tool calls, tokens and latency | needs two finished real-model experiments |

## Honest limits

- **D is scripted.** Its agent is a deliberate fake that claims success. The demo says so. If a real
  experiment produces the same failure, show that run in the experiment screen instead (run detail
  → *refund failures*).
- **C's restart is simulated.** After the run, the demo puts the refund operation back to `unknown`,
  as a crash leaves it, and runs the same reconciliation step the start-up sweep uses. The real
  kill mid-flight is `tests/failure_scenarios::test_worker_crash_resumes_from_persisted_state`.
- **E is not run yet.** It compares two real-model configurations, so it waits for the W5.2
  experiments (`uv run python -m scripts.dev evaluate --plan main`, about 1,200 free-tier requests,
  resumed over several days). Then:
  `uv run python -m scripts.dev demo e --baseline EXP --candidate EXP` (the main plan holds both
  configurations in one experiment, so pass its ID twice). The comparison is the same one the
  `/ui/compare` screen shows.

## Screens to open next to the demos

`uv run uvicorn evals.api:app --port 8100` on `support_eval` (set `DATABASE_URL` to it, or use the
dev commands), `uv run python -m scripts.mint_token auth|ops --role operator`, then
`/ui/experiments`. The agent's own screens are `/demo` and `/review` on the main app.
