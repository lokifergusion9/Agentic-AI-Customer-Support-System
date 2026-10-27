"""Chunk, embed and store the synthetic policy documents (W2.1).

    uv run python -m scripts.ingest_policies

Each `## Section N` of a document becomes one row in policy_chunks, keyed by its citation
(`refund_policy_v1:section_3`). Running it again is safe: a document whose file has not changed
(same content hash) is skipped, and a changed one is replaced with all of its chunks. The first
run downloads the embedding model (about 90 MB) into the Hugging Face cache.
"""

import json
from dataclasses import dataclass

from sqlalchemy import Connection, create_engine, text

from app.config import get_settings
from app.db import SCHEMA
from app.policies.documents import PolicyDocument, PolicySection, load_policies
from app.policies.embeddings import Embedder, get_embedder, vector_literal


@dataclass(frozen=True)
class IngestResult:
    inserted: list[str]
    replaced: list[str]
    unchanged: list[str]


def embedding_text(document: PolicyDocument, section: PolicySection) -> str:
    """The title and heading carry meaning the section body often leaves implicit."""
    return f"{document.title}. Section {section.number}: {section.title}.\n{section.content}"


def ingest(
    connection: Connection, embedder: Embedder, documents: list[PolicyDocument]
) -> IngestResult:
    """Store `documents` in the caller's transaction."""
    stored = dict(
        connection.execute(text(f"SELECT id, content_hash FROM {SCHEMA}.policy_documents")).all()
    )
    result = IngestResult([], [], [])
    for document in documents:
        previous = stored.get(document.document_id)
        if previous == document.content_hash:
            result.unchanged.append(document.document_id)
            continue
        if previous is not None:
            # Chunks go with it (ON DELETE CASCADE).
            connection.execute(
                text(f"DELETE FROM {SCHEMA}.policy_documents WHERE id = :id"),
                {"id": document.document_id},
            )
            result.replaced.append(document.document_id)
        else:
            result.inserted.append(document.document_id)
        store_document(connection, embedder, document)
    return result


def store_document(connection: Connection, embedder: Embedder, document: PolicyDocument) -> None:
    connection.execute(
        text(
            f"INSERT INTO {SCHEMA}.policy_documents (id, policy_key, version, title, "
            "effective_from, effective_to, source_uri, content_hash) "
            "VALUES (:id, :key, :version, :title, :from_, :to, :uri, :hash)"
        ),
        {
            "id": document.document_id,
            "key": document.policy_key,
            "version": document.version,
            "title": document.title,
            "from_": document.effective_from,
            "to": document.effective_to,
            "uri": document.source_uri,
            "hash": document.content_hash,
        },
    )
    vectors = embedder.embed([embedding_text(document, s) for s in document.sections])
    for section, vector in zip(document.sections, vectors, strict=True):
        if len(vector) != embedder.dimensions:
            raise ValueError(f"expected {embedder.dimensions} dimensions, got {len(vector)}")
        metadata = {
            "policy_key": document.policy_key,
            "version": document.version,
            "document_title": document.title,
            "effective_from": document.effective_from.isoformat(),
            "effective_to": document.effective_to and document.effective_to.isoformat(),
        }
        connection.execute(
            text(
                f"INSERT INTO {SCHEMA}.policy_chunks (id, document_id, section_number, "
                "section_title, content, metadata, embedding) VALUES (:id, :document, :number, "
                ":title, :content, CAST(:metadata AS jsonb), CAST(:embedding AS public.vector))"
            ),
            {
                "id": document.citation(section.number),
                "document": document.document_id,
                "number": section.number,
                "title": section.title,
                "content": section.content,
                "metadata": json.dumps(metadata),
                "embedding": vector_literal(vector),
            },
        )


def main() -> None:
    engine = create_engine(get_settings().database_url)
    with engine.begin() as connection:
        result = ingest(connection, get_embedder(), load_policies())
    engine.dispose()
    print(
        f"policies: {len(result.inserted)} inserted, {len(result.replaced)} replaced, "
        f"{len(result.unchanged)} unchanged"
    )


if __name__ == "__main__":
    main()
