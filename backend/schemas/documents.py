from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    upload_timestamp: datetime = Field(default_factory=utcnow)
    num_pages: int = 0
    status: DocumentStatus = DocumentStatus.PROCESSING
    error: str | None = None
    num_chunks: int = 0
    file_size_bytes: int = 0
    empty_pages: list[int] = Field(default_factory=list)
    embedding_model: str | None = None


class PageText(BaseModel):
    """Raw text extracted from one PDF page (1-based page number)."""
    page_number: int
    text: str


class ExtractedDocument(BaseModel):
    num_pages: int
    pages: list[PageText]
    empty_pages: list[int] = Field(default_factory=list)
    repaired: bool = False


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_start: int
    page_end: int
    pages: list[int]
    text: str

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


UploadOutcome = Literal["indexed", "duplicate", "failed", "rejected"]


class UploadResult(BaseModel):
    filename: str
    status: UploadOutcome
    message: str
    document_id: str | None = None
    num_pages: int | None = None
    num_chunks: int | None = None


class UploadResponse(BaseModel):
    results: list[UploadResult]


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]
    total: int


class DeleteResponse(BaseModel):
    deleted: str
    chunks_removed: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    vector_db: str
    indexed_documents: int
    failed_documents: int
    total_chunks: int
    embedding_model: str
    embedding_model_loaded: bool
    llm_provider: str
    llm_model: str
    llm_configured: bool
