"""Verified facts and citations, taken by code from successful tool results (W2.6, W2.12).

The model's summaries are never the evidence. A proposal may only cite record IDs that a read
tool actually returned in this run, and policy citations that search_policy returned.
"""

from collections.abc import Iterable
from typing import Any

from app.agents.loop import ToolRecord
from app.tools.registry import READ_TOOLS

REF_KEYS = {
    "id",
    "order_id",
    "return_id",
    "payment_id",
    "operation_id",
    "provider_reference",
    "tracking_id",
}


def fact_key(record: ToolRecord) -> str:
    argument = (
        record.arguments.get("order_id")
        or record.arguments.get("operation_id")
        or record.arguments.get("ticket_id")
    )
    return f"{record.name}:{argument}" if argument else record.name


def collect_facts(records: Iterable[ToolRecord]) -> dict[str, Any]:
    """Successful read results, keyed by tool and argument; search results are kept apart."""
    return {
        fact_key(record): record.result["data"]
        for record in records
        if record.result.get("success")
        and record.name in READ_TOOLS
        and record.name != "search_policy"
    }


def collect_citations(records: Iterable[ToolRecord]) -> list[str]:
    citations: list[str] = []
    for record in records:
        if record.name == "search_policy" and record.result.get("success"):
            for citation in record.result["data"]["citations"]:
                if citation not in citations:
                    citations.append(citation)
    return citations


def known_refs(facts: dict[str, Any]) -> set[str]:
    """Every record ID that appears anywhere in the verified facts."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in REF_KEYS and isinstance(item, str):
                    found.add(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(facts)
    return found


def verified_order(facts: dict[str, Any], preferred: str | None = None) -> dict[str, Any] | None:
    """The order get_order returned for `preferred`, else the only order looked up.

    get_order succeeds only for the customer's own orders, so a returned order is owned.
    """
    if preferred and f"get_order:{preferred}" in facts:
        return facts[f"get_order:{preferred}"]
    orders = [value for key, value in facts.items() if key.startswith("get_order:")]
    return orders[0] if len(orders) == 1 else None
