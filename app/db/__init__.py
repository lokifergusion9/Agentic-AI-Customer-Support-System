"""Database models, sessions and LangGraph checkpointer config."""

# Application tables live in this PostgreSQL schema. LangGraph checkpoint tables use a
# separate schema (W3.3), so the two never mix (W1.4).
SCHEMA = "support"
