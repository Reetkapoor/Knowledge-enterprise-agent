from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from backend.schemas.agentic import MechanismFlags, PipelineMode, VerificationResult


class ChatRequest(BaseModel):
    question: str = Field(..., description="Natural-language question")
    session_id: str | None = Field(None, description="Conversation id; generated if omitted")
    top_k: int | None = Field(None, ge=1, le=20)
    document_ids: list[str] | None = Field(None, description="Restrict retrieval to these documents")
    mode: PipelineMode = Field(PipelineMode.BASELINE, description="baseline | full | custom")
    flags: MechanismFlags | None = Field(None, description="Per-mechanism switches; used when mode=custom")

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty")
        if len(v) > 2000:
            raise ValueError("Question is too long (max 2000 characters)")
        return v


class RetrievedChunk(BaseModel):
    """One retrieved passage with its provenance and score. Built from stored metadata only."""
    rank: int
    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    page_label: str
    text: str
    similarity: float = Field(..., description="Cosine similarity (1 - cosine distance)")
    distance: float
    cited: bool = False


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    candidates_considered: int = 0
    dropped_below_threshold: int = 0


class Citation(BaseModel):
    source_id: str = Field(..., description="Marker the LLM used, e.g. S1")
    document_id: str
    filename: str
    page_start: int
    page_end: int
    page_label: str
    chunk_id: str
    passage: str


class Turn(BaseModel):
    question: str
    answer: str


class ProcessedQuery(BaseModel):
    """Output of query processing. Adaptive RAG can later add fields (complexity, sub-queries)."""
    original: str
    standalone: str
    rewritten: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    question: str
    standalone_question: str
    answer: str
    status: Literal["answered", "insufficient_information", "conversational"]
    reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[RetrievedChunk] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    mode: str = "baseline"
    mechanisms: MechanismFlags = Field(default_factory=MechanismFlags)
    original_answer: str | None = Field(None, description="Pre-verification answer (when verification ran)")
    verification: VerificationResult | None = None
