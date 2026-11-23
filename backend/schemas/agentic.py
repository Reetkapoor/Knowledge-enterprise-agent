"""Schemas shared by Adaptive RAG, Corrective RAG and claim-level verification."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PipelineMode(str, Enum):
    BASELINE = "baseline"
    FULL = "full"
    CUSTOM = "custom"  # explicit per-mechanism flags (ablations)


class MechanismFlags(BaseModel):
    adaptive: bool = False
    corrective: bool = False
    verification: bool = False

    @classmethod
    def for_mode(cls, mode: PipelineMode, custom: "MechanismFlags | None" = None) -> "MechanismFlags":
        if mode == PipelineMode.FULL:
            return cls(adaptive=True, corrective=True, verification=True)
        if mode == PipelineMode.CUSTOM and custom is not None:
            return custom
        return cls()

    @property
    def label(self) -> str:
        on = [n for n in ("adaptive", "corrective", "verification") if getattr(self, n)]
        return "baseline" if not on else "+".join(on)


# ------------------------------------------------------------------ Adaptive RAG
class QueryType(str, Enum):
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"
    CONVERSATIONAL = "CONVERSATIONAL"  # greeting / thanks / questions about the assistant itself


class RoutePlan(BaseModel):
    query_type: QueryType
    sub_questions: list[str] = Field(default_factory=list)
    reasoning: str = ""
    conversational_reply: str | None = None
    error: str | None = None  # set when routing failed and the default (SIMPLE) plan was used


# ------------------------------------------------------------------ Corrective RAG
class RetrievalVerdict(str, Enum):
    RELEVANT = "RELEVANT"
    PARTIALLY_RELEVANT = "PARTIALLY_RELEVANT"
    IRRELEVANT = "IRRELEVANT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class RetrievalEvaluation(BaseModel):
    verdict: RetrievalVerdict
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    sufficient: bool = False
    missing_information: str = ""
    reasoning: str = ""
    error: str | None = None


class RetrievalAttempt(BaseModel):
    query: str
    chunk_ids: list[str]
    pages: list[str]  # "filename:page_label" for compact traces
    evaluation: RetrievalEvaluation | None = None


class CorrectiveTrace(BaseModel):
    """Everything needed to explain why retrieval was (not) retried."""
    question: str
    attempts: list[RetrievalAttempt] = Field(default_factory=list)
    retried: bool = False
    retry_reason: str | None = None
    refined_query: str | None = None
    final_verdict: RetrievalVerdict | None = None


# ------------------------------------------------------------------ Verification
class ClaimVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class Claim(BaseModel):
    claim_id: str
    text: str
    cited_sources: list[str] = Field(default_factory=list)  # markers the generator attached, e.g. ["S1"]


class EvidenceMatch(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    page_label: str
    window_text: str
    similarity: float
    entailment: float
    neutral: float
    contradiction: float


class CitationCheck(BaseModel):
    source_id: str
    chunk_id: str | None = None
    exists: bool = False
    page_valid: bool = False
    supports_claim: bool = False
    reason: str = ""

    @property
    def valid(self) -> bool:
        return self.exists and self.page_valid and self.supports_claim


class VerifiedClaim(BaseModel):
    claim: Claim
    verdict: ClaimVerdict
    entailment: float = 0.0
    contradiction: float = 0.0
    conflict: bool = False  # both entailing and contradicting evidence above thresholds
    demoted_by_numeric_guard: bool = False
    supporting: EvidenceMatch | None = None
    contradicting: EvidenceMatch | None = None
    evidence: list[EvidenceMatch] = Field(default_factory=list)
    citation_checks: list[CitationCheck] = Field(default_factory=list)
    final_citations: list[str] = Field(default_factory=list)  # chunk_ids that validly support the claim
    reattributed: bool = False  # citation replaced by the best supporting retrieved chunk
    action: str = ""  # kept | removed | corrected | qualified


class VerificationResult(BaseModel):
    enabled: bool = True
    original_answer: str
    final_answer: str
    claims: list[VerifiedClaim] = Field(default_factory=list)
    unavailable_statements: list[str] = Field(default_factory=list)
    abstained: bool = False
    counts: dict[str, int] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    nli_model: str = ""
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
