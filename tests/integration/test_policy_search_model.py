"""Retrieval quality with the real embedding model (W2.2). Opt-in: `pytest -m embedding_model`.

Each question is asked the way a customer or the policy agent would. An expected citation must
be in the top two passages; an expected document ID (no section) must supply the top passage.
This is also what MIN_RELEVANCE in app/policies/search.py is tuned against.
"""

from collections.abc import Iterator
from datetime import date

import pytest
from sqlalchemy import Connection
from sqlalchemy.orm import Session

from app.policies.documents import load_policies
from app.policies.embeddings import get_embedder
from app.policies.search import PolicyFilters, build_policy_query, search_policy
from scripts.ingest_policies import ingest

pytestmark = pytest.mark.embedding_model

ON = date(2026, 10, 20)


@pytest.fixture
def db(conn: Connection) -> Iterator[Session]:
    ingest(conn, get_embedder(), load_policies())
    with Session(bind=conn) as session:
        yield session


@pytest.mark.parametrize(
    ("question", "facts", "expected"),
    [
        (
            "I returned my headphones last week, but I still haven't received my refund.",
            {"return_status": "received", "inspection_status": "passed"},
            "refund_policy_v1:section_3",
        ),
        (
            "How long does the money take to reach my bank after a refund?",
            None,
            "refund_policy_v1:section_6",
        ),
        ("My new laptop arrived with a cracked screen.", None, "damaged_products_v1"),
        (
            "What details do I need to give for a damaged product claim?",
            None,
            "damaged_products_v1:section_3",
        ),
        ("My order is late and still hasn't arrived.", None, "shipping_delays_v1:section_4"),
        (
            "The inspection of my returned item failed, what happens now?",
            None,
            "return_processing_v1:section_5",
        ),
        (
            "A refund of 90,000 rupees, who has to approve it?",
            None,
            "human_review_exceptions_v1:section_3",
        ),
    ],
)
def test_expected_policy_ranks_at_the_top(
    db: Session, question: str, facts: dict | None, expected: str
) -> None:
    result = search_policy(
        db, get_embedder(), build_policy_query(question, facts), PolicyFilters(effective_on=ON)
    )
    assert result.status == "found", result.message
    if ":" in expected:
        assert expected in result.citations[:2], result.citations
    else:
        assert result.passages[0].document_id == expected, result.citations


def test_unrelated_question_is_below_the_relevance_threshold(db: Session) -> None:
    result = search_policy(
        db, get_embedder(), "What is the boiling point of nitrogen?", PolicyFilters(effective_on=ON)
    )
    assert result.status == "no_relevant_passage"
