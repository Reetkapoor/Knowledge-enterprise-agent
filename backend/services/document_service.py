"""Ingestion orchestration: PDF -> text -> chunks -> embeddings -> vector store.

Runs at upload time only; question answering never touches this path.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass

from backend.core.config import Settings
from backend.schemas.documents import DocumentRecord, DocumentStatus
from backend.services.document_registry import DocumentRegistry
from backend.services.embedding_service import BaseEmbedder, EmbeddingError
from backend.services.pdf_processor import PDFProcessingError, extract_pdf
from backend.services.text_chunker import chunk_pages
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    record: DocumentRecord
    outcome: str  # indexed | duplicate | failed
    message: str


class DocumentService:
    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry,
        vector_store: VectorStore,
        embedder: BaseEmbedder,
    ):
        self.settings = settings
        self.registry = registry
        self.vector_store = vector_store
        self.embedder = embedder
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # one ingest/delete at a time keeps index + registry consistent

    @staticmethod
    def compute_document_id(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()[:16]

    def ingest(self, filename: str, content: bytes) -> IngestResult:
        document_id = self.compute_document_id(content)
        with self._lock:
            existing = self.registry.get(document_id)
            if existing and existing.status == DocumentStatus.INDEXED:
                return IngestResult(
                    existing,
                    "duplicate",
                    f"Identical content is already indexed as '{existing.filename}'; nothing was re-added.",
                )

            record = DocumentRecord(
                document_id=document_id,
                filename=filename,
                file_size_bytes=len(content),
                embedding_model=self.embedder.model_name,
            )
            self.registry.upsert(record)
            try:
                extracted = extract_pdf(content)
                record.num_pages = extracted.num_pages
                record.empty_pages = extracted.empty_pages

                chunks = chunk_pages(
                    extracted.pages,
                    document_id=document_id,
                    filename=filename,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                    tokens_per_word=self.settings.tokens_per_word,
                )
                # Skip chunks already in the index (e.g. partial earlier attempt).
                present = self.vector_store.existing_ids([c.chunk_id for c in chunks])
                new_chunks = [c for c in chunks if c.chunk_id not in present]
                if new_chunks:
                    vectors = self.embedder.embed_documents([c.text for c in new_chunks])
                    self.vector_store.add_chunks(new_chunks, vectors, self.embedder.model_name)

                (self.settings.upload_dir / f"{document_id}.pdf").write_bytes(content)
                record.num_chunks = len(chunks)
                record.status = DocumentStatus.INDEXED
                record.error = None
                self.registry.upsert(record)
                note = f" ({len(extracted.empty_pages)} page(s) had no text)" if extracted.empty_pages else ""
                if extracted.repaired:
                    note += " Warning: the PDF was damaged and auto-repaired; some content may be missing."
                return IngestResult(
                    record, "indexed", f"Indexed {len(chunks)} chunks from {extracted.num_pages} pages{note}."
                )
            except (PDFProcessingError, EmbeddingError) as exc:
                return self._fail(record, str(exc))
            except Exception as exc:
                logger.exception("Unexpected ingestion failure for %s", filename)
                return self._fail(record, f"Unexpected error while processing ({type(exc).__name__}).")

    def _fail(self, record: DocumentRecord, reason: str) -> IngestResult:
        logger.warning("Ingestion failed for %s: %s", record.filename, reason)
        try:
            self.vector_store.delete_document(record.document_id)  # no partial index
        except Exception:
            logger.exception("Cleanup after failed ingestion also failed")
        record.status, record.error, record.num_chunks = DocumentStatus.FAILED, reason, 0
        self.registry.upsert(record)
        return IngestResult(record, "failed", reason)

    def delete(self, document_id: str) -> int | None:
        """Returns number of chunks removed, or None if the document is unknown."""
        with self._lock:
            if self.registry.get(document_id) is None:
                return None
            removed = self.vector_store.delete_document(document_id)
            (self.settings.upload_dir / f"{document_id}.pdf").unlink(missing_ok=True)
            self.registry.delete(document_id)
            return removed
