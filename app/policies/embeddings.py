"""Text embeddings for policy retrieval (Q3: local sentence-transformers, 384 dimensions).

The model runs locally, so there is no API key and no per-call cost, and the same text always
gets the same vector. Tests use a small deterministic fake instead of downloading the model.
"""

from functools import lru_cache
from typing import Protocol

from app.config import get_settings


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    def __init__(self, model_name: str, dimensions: int) -> None:
        # Imported here: torch is slow to import and only ingestion and search need it.
        from sentence_transformers import SentenceTransformer

        try:
            # Once downloaded, load from the local cache: no network call, no Hub rate limits.
            self.model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        except OSError:
            self.model = SentenceTransformer(model_name, device="cpu")
        self.dimensions = dimensions
        actual = self.model.get_embedding_dimension()
        if actual != dimensions:
            raise ValueError(
                f"{model_name} produces {actual}-dimensional vectors, "
                f"but EMBEDDING_DIMENSIONS is {dimensions}"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Normalized, so cosine distance in pgvector equals 1 - dot product.
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return LocalEmbedder(settings.embedding_model, settings.embedding_dimensions)


def vector_literal(vector: list[float]) -> str:
    """pgvector's text form, for `CAST(:embedding AS public.vector)`."""
    return "[" + ",".join(f"{value:.7g}" for value in vector) + "]"
