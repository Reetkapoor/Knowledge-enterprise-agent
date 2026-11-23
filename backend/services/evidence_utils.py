"""Small helpers for combining retrieved chunks from several searches."""
from __future__ import annotations

from backend.schemas.chat import RetrievedChunk


def dedupe_best(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Unique by chunk_id (keeping the highest similarity), sorted best-first, ranks re-assigned."""
    best: dict[str, RetrievedChunk] = {}
    for c in chunks:
        if c.chunk_id not in best or c.similarity > best[c.chunk_id].similarity:
            best[c.chunk_id] = c
    out = sorted(best.values(), key=lambda c: -c.similarity)
    return _rerank([c.model_copy() for c in out])


def round_robin(groups: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Interleave per-sub-question result lists so every sub-question keeps its best evidence
    when the generator's context budget cuts the tail."""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for i in range(max((len(g) for g in groups), default=0)):
        for g in groups:
            if i < len(g) and g[i].chunk_id not in seen:
                seen.add(g[i].chunk_id)
                out.append(g[i].model_copy())
    return _rerank(out)


def page_ref(c: RetrievedChunk) -> str:
    return f"{c.filename}:{c.page_label}"


def _rerank(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    for rank, c in enumerate(chunks, start=1):
        c.rank = rank
    return chunks
