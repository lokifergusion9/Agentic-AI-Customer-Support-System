"""Policy ingestion into policy_documents / policy_chunks (W2.1)."""

from dataclasses import replace

import pytest
from sqlalchemy import Connection, text

from app.db import SCHEMA
from app.policies.documents import load_policies
from scripts.ingest_policies import ingest
from tests.fakes import FakeEmbedder


def chunk_ids(conn: Connection) -> list[str]:
    return list(conn.execute(text(f"SELECT id FROM {SCHEMA}.policy_chunks ORDER BY id")).scalars())


def test_ingest_stores_every_section_with_its_citation(conn: Connection) -> None:
    documents = load_policies()
    result = ingest(conn, FakeEmbedder(), documents)

    assert sorted(result.inserted) == [d.document_id for d in documents]
    assert len(chunk_ids(conn)) == sum(len(d.sections) for d in documents)
    row = conn.execute(
        text(
            "SELECT c.section_title, c.content, c.metadata, d.effective_from, "
            "vector_dims(c.embedding) AS dims "
            f"FROM {SCHEMA}.policy_chunks c JOIN {SCHEMA}.policy_documents d "
            "ON d.id = c.document_id WHERE c.id = 'refund_policy_v1:section_3'"
        )
    ).one()
    assert row.section_title == "Eligibility"
    assert "inspection passes" in row.content
    assert row.metadata["policy_key"] == "refund_policy"
    assert row.metadata["effective_to"] is None
    assert str(row.effective_from) == "2026-01-01"
    assert row.dims == 384


def test_rerun_skips_unchanged_documents(conn: Connection) -> None:
    documents = load_policies()
    ingest(conn, FakeEmbedder(), documents)
    embedder = FakeEmbedder()

    result = ingest(conn, embedder, documents)

    assert result.inserted == [] and result.replaced == []
    assert len(result.unchanged) == len(documents)
    assert embedder.calls == 0


def test_changed_document_is_replaced_with_its_chunks(conn: Connection) -> None:
    documents = load_policies()
    ingest(conn, FakeEmbedder(), documents)
    refund = next(d for d in documents if d.document_id == "refund_policy_v1")
    shorter = replace(refund, content_hash="changed", sections=refund.sections[:2])

    result = ingest(conn, FakeEmbedder(), [shorter])

    assert result.replaced == ["refund_policy_v1"]
    remaining = [c for c in chunk_ids(conn) if c.startswith("refund_policy_v1:")]
    assert remaining == ["refund_policy_v1:section_1", "refund_policy_v1:section_2"]


class ShortVectorEmbedder(FakeEmbedder):
    def vector(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def test_wrong_embedding_size_is_rejected(conn: Connection) -> None:
    with pytest.raises(ValueError, match="expected 384 dimensions, got 3"):
        ingest(conn, ShortVectorEmbedder(), load_policies()[:1])
