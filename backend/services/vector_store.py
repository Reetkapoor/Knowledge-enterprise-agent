"""ChromaDB wrapper: persistent storage, cosine search, per-document delete."""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from backend.schemas.chat import RetrievedChunk
from backend.schemas.documents import Chunk

logger = logging.getLogger(__name__)
_BATCH = 500


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "enterprise_documents"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ---- writes -------------------------------------------------------------
    def existing_ids(self, ids: list[str]) -> set[str]:
        found: set[str] = set()
        for i in range(0, len(ids), _BATCH):
            found.update(self._col.get(ids=ids[i : i + _BATCH], include=[])["ids"])
        return found

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray, embedding_model: str = "") -> int:
        """Upsert chunks (idempotent on chunk_id, so re-adding never duplicates)."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        for i in range(0, len(chunks), _BATCH):
            part = chunks[i : i + _BATCH]
            self._col.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=embeddings[i : i + _BATCH].tolist(),
                documents=[c.text for c in part],
                metadatas=[
                    {
                        "document_id": c.document_id,
                        "filename": c.filename,
                        "chunk_index": c.chunk_index,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "pages": ",".join(map(str, c.pages)),
                        "embedding_model": embedding_model,
                    }
                    for c in part
                ],
            )
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        existing = self._col.get(where={"document_id": document_id}, include=[])["ids"]
        if existing:
            self._col.delete(ids=existing)
        return len(existing)

    # ---- reads --------------------------------------------------------------
    def count(self) -> int:
        return self._col.count()

    def query(
        self, embedding: list[float], top_k: int, document_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        total = self._col.count()
        if total == 0:
            return []
        where = None
        if document_ids:
            where = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": list(document_ids)}}
            )
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        for rank, (cid, text, meta, dist) in enumerate(
            zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
        ):
            ps, pe = int(meta["page_start"]), int(meta["page_end"])
            out.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=cid,
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    page_start=ps,
                    page_end=pe,
                    page_label=str(ps) if ps == pe else f"{ps}-{pe}",
                    text=text,
                    similarity=round(1.0 - float(dist), 4),
                    distance=round(float(dist), 4),
                )
            )
        return out

    def health(self) -> str:
        try:
            self._client.heartbeat()
            return "ok"
        except Exception as exc:
            return f"error: {type(exc).__name__}"
