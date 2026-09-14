"""Local embedding models using sentence-transformers with caching and test fakes."""

from __future__ import annotations

import math
from typing import Protocol


class Embeddings(Protocol):
    """Protocol defining the embedding interface."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return normalized float vectors."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text and return a normalized float vector."""
        ...

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality of this embedding model."""
        ...


class SentenceTransformerEmbeddings:
    """Wrapper around sentence-transformers with lazy loading and normalization."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dimension: int | None = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            model = self._get_model()
            dim = model.get_sentence_embedding_dimension()
            self._dimension = int(dim) if dim is not None else 384
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        results = self.embed_texts([text])
        if not results:
            return [0.0] * self.dimension
        return results[0]


class FakeEmbeddings:
    """Deterministic in-memory embeddings for tests without network or heavy models."""

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dimension
            seed = sum(ord(c) for c in text) or 1
            for i in range(self._dimension):
                vec[i] = float(((i + 1) * seed) % 100 + 1)
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([x / norm for x in vec])
        return results

    def embed_query(self, text: str) -> list[float]:
        results = self.embed_texts([text])
        if not results:
            return [0.0] * self._dimension
        return results[0]


_EMBEDDINGS_CACHE: dict[str, SentenceTransformerEmbeddings] = {}


def get_embedding_generator(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformerEmbeddings:
    """Return a cached instance of the requested sentence-transformer model."""
    if model_name not in _EMBEDDINGS_CACHE:
        _EMBEDDINGS_CACHE[model_name] = SentenceTransformerEmbeddings(model_name)
    return _EMBEDDINGS_CACHE[model_name]
