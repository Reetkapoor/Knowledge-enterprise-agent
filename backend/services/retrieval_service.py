"""Dense-vector retrieval. Knows nothing about LLMs or answer generation."""
from __future__ import annotations

from backend.schemas.chat import RetrievalResult
from backend.services.embedding_service import BaseEmbedder
from backend.services.vector_store import VectorStore


class RetrievalService:
    def __init__(self, embedder: BaseEmbedder, vector_store: VectorStore, top_k: int, min_similarity: float):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(
        self, query: str, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> RetrievalResult:
        candidates = self.vector_store.query(
            self.embedder.embed_query(query), top_k or self.top_k, document_ids
        )
        kept = [c for c in candidates if c.similarity >= self.min_similarity]
        for rank, chunk in enumerate(kept, start=1):
            chunk.rank = rank
        return RetrievalResult(
            query=query,
            chunks=kept,
            candidates_considered=len(candidates),
            dropped_below_threshold=len(candidates) - len(kept),
        )
