"""Deterministic stand-ins for slow or paid dependencies, shared by unit and integration tests."""

import hashlib
import math
import re


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors, with no model.

    Good enough to test ranking and filtering logic; real semantic quality is tested only in
    the `embedding_model` marked tests.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self.vector(text) for text in texts]

    def vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
