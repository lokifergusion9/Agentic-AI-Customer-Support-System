"""The evaluation platform's PostgreSQL schema (W4.3, Q8).

Separate from the `support` schema the agent reads, so expected outcomes never sit where the
agent's tools look. Code under `app/` never refers to it.
"""

EVAL_SCHEMA = "eval"
