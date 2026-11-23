"""Corrective RAG: judge retrieval quality, and if it is poor, retry once with a refined query.

The evaluator only looks at *retrieved passages vs. the question*. It never sees a generated
answer; judging answers is the verification module's job.

Verdict rules (deterministic, applied to the LLM's per-passage judgements):
  no passage judged relevant                              -> IRRELEVANT
  some relevant but evidence not sufficient               -> INSUFFICIENT_EVIDENCE
  sufficient, but <= partial_threshold of passages relevant (noisy) -> PARTIALLY_RELEVANT
  sufficient and mostly relevant                          -> RELEVANT
Retry happens for the verdicts in `retry_on` (default: IRRELEVANT, INSUFFICIENT_EVIDENCE).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from backend.schemas.agentic import (
    CorrectiveTrace,
    RetrievalAttempt,
    RetrievalEvaluation,
    RetrievalVerdict,
)
from backend.schemas.chat import RetrievedChunk
from backend.services.evidence_utils import dedupe_best, page_ref
from backend.services.llm_json import StructuredOutputError, generate_structured
from backend.services.llm_service import LLMError, LLMProvider
from backend.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

_EVAL_SYSTEM = """You judge document retrieval quality for a question-answering system.
You get a QUESTION and numbered retrieved PASSAGES. Judge ONLY the passages; do not answer the question.

1. Identify passages containing information relevant to the question.
2. Decide whether the relevant passages TOGETHER provide enough information
   to answer the question accurately.
3. Evaluate sufficiency based on the information actually present.
   Do not assume that additional requirements exist unless the passages
   indicate that essential information is missing.
4. If a numbered policy section contains its complete list of conditions,
   treat those conditions as sufficient when they answer the question.
5. Do not confuse requirements from different scholarships, programs,
   policies, or academic years.
6. If the evidence supports a partial answer but an essential requested
   detail is missing, set sufficient to false and identify that detail.
7. Do not treat an unrelated section as evidence that the requested
   section is incomplete.

Reply with one JSON object and nothing else:
{"relevant_passages": [<numbers of relevant passages>], "sufficient": true|false, "missing_information": "<short, empty if sufficient>", "reasoning": "<one short sentence>"}"""

_REFINE_SYSTEM = """You rewrite a failed document search query so a retry finds the missing information.
You get the ORIGINAL QUESTION, the QUERY that failed, what was MISSING, and the passages that came back.
Write ONE new search query: use different or more specific wording, likely terminology from a policy/handbook
document, and target the missing information. Do not answer the question or add facts.
Reply with one JSON object and nothing else: {"refined_query": "<query>"}"""


class _EvalOut(BaseModel):
    relevant_passages: list[int] = Field(default_factory=list)
    sufficient: bool = False
    missing_information: str = ""
    reasoning: str = ""


class _RefineOut(BaseModel):
    refined_query: str


def derive_verdict(n_passages: int, n_relevant: int, sufficient: bool, partial_threshold: float) -> RetrievalVerdict:
    if n_passages == 0 or n_relevant == 0:
        return RetrievalVerdict.IRRELEVANT
    if not sufficient:
        return RetrievalVerdict.INSUFFICIENT_EVIDENCE
    if n_relevant / n_passages <= partial_threshold:
        return RetrievalVerdict.PARTIALLY_RELEVANT
    return RetrievalVerdict.RELEVANT


class RetrievalEvaluator:
    def __init__(self, llm: LLMProvider, partial_threshold: float = 0.5, passage_chars: int = 4000):
        self.llm = llm
        self.partial_threshold = partial_threshold
        self.passage_chars = passage_chars

    def evaluate(self, question: str, chunks: list[RetrievedChunk]) -> RetrievalEvaluation:
        if not chunks:
            return RetrievalEvaluation(
                verdict=RetrievalVerdict.IRRELEVANT, missing_information=question, reasoning="no passages retrieved"
            )
        blocks = [f"[{i}] ({c.filename}, page {c.page_label})\n{c.text[: self.passage_chars]}" for i, c in enumerate(chunks, 1)]
        prompt = f"QUESTION: {question}\n\nPASSAGES:\n\n" + "\n\n".join(blocks)
        try:
            out = generate_structured(self.llm, _EVAL_SYSTEM, prompt, _EvalOut, max_tokens=300)
        except LLMError as exc:
            if exc.not_configured:
                raise
            return self._unavailable(f"evaluator LLM error: {exc}")
        except StructuredOutputError as exc:
            return self._unavailable(f"evaluator output invalid: {exc}")
        idx = sorted({n for n in out.relevant_passages if 1 <= n <= len(chunks)})
        verdict = derive_verdict(len(chunks), len(idx), out.sufficient, self.partial_threshold)
        return RetrievalEvaluation(
            verdict=verdict,
            relevant_chunk_ids=[chunks[n - 1].chunk_id for n in idx],
            sufficient=out.sufficient and bool(idx),
            missing_information=out.missing_information.strip(),
            reasoning=out.reasoning.strip(),
        )

    @staticmethod
    def _unavailable(error: str) -> RetrievalEvaluation:
        # Evaluation failed: pass retrieval through unchanged, but say so explicitly.
        logger.warning("Retrieval evaluation unavailable: %s", error)
        return RetrievalEvaluation(
            verdict=RetrievalVerdict.RELEVANT, reasoning="evaluation unavailable; retrieval passed through unchanged", error=error
        )


class QueryRefiner:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def refine(self, question: str, failed_query: str, evaluation: RetrievalEvaluation, chunks: list[RetrievedChunk]) -> str:
        seen = "\n".join(f"- ({c.filename}, p.{c.page_label}) {c.text[:160]!r}" for c in chunks[:5]) or "- (nothing retrieved)"
        missing = evaluation.missing_information or "unspecified"
        prompt = (
            f"ORIGINAL QUESTION: {question}\nFAILED QUERY: {failed_query}\nMISSING: {missing}\n"
            f"PASSAGES THAT CAME BACK:\n{seen}"
        )
        try:
            refined = " ".join(generate_structured(self.llm, _REFINE_SYSTEM, prompt, _RefineOut, max_tokens=120).refined_query.split())
        except LLMError as exc:
            if exc.not_configured:
                raise
            refined = ""
        except StructuredOutputError:
            refined = ""
        if not refined or refined.lower() == failed_query.lower() or len(refined) > 400:
            refined = f"{question} {evaluation.missing_information}".strip()  # deterministic fallback
        return refined


@dataclass
class CorrectiveOutcome:
    chunks: list[RetrievedChunk]
    trace: CorrectiveTrace
    retrieval_calls: int = 1
    verdict_before: RetrievalVerdict | None = None
    verdict_after: RetrievalVerdict | None = None
    chunks_before: list[RetrievedChunk] = field(default_factory=list)  # first-attempt chunks, for evaluation
    evaluation: RetrievalEvaluation | None = None  # final evaluation


class CorrectiveRetriever:
    def __init__(
        self, retriever: RetrievalService, evaluator: RetrievalEvaluator, refiner: QueryRefiner,
        max_retries: int = 1, retry_on: set[str] | None = None,
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.refiner = refiner
        self.max_retries = max(0, max_retries)
        self.retry_on = retry_on or {RetrievalVerdict.IRRELEVANT.value, RetrievalVerdict.INSUFFICIENT_EVIDENCE.value}

    def run(self, query: str, top_k: int | None = None, document_ids: list[str] | None = None) -> CorrectiveOutcome:
        trace = CorrectiveTrace(question=query)
        first = self.retriever.retrieve(query, top_k, document_ids)
        calls = 1
        current_query, chunks = query, first.chunks
        evaluation = self.evaluator.evaluate(query, chunks)
        trace.attempts.append(_attempt(current_query, chunks, evaluation))
        verdict_before = evaluation.verdict
        retries = 0

        while evaluation.verdict.value in self.retry_on and retries < self.max_retries and evaluation.error is None:
            retries += 1
            if retries == 1:
                trace.retried = True
                trace.retry_reason = (
                    f"{evaluation.verdict.value}: {evaluation.reasoning or 'n/a'}"
                    + (f" Missing: {evaluation.missing_information}" if evaluation.missing_information else "")
                )
            refined = self.refiner.refine(query, current_query, evaluation, chunks)
            trace.refined_query = refined
            retry = self.retriever.retrieve(refined, top_k, document_ids)
            calls += 1
            kept_before = [c for c in chunks if c.chunk_id in set(evaluation.relevant_chunk_ids)]
            chunks = dedupe_best(retry.chunks + kept_before)  # keep what was already useful
            current_query = refined
            # judged against the ORIGINAL question, on the combined evidence
            evaluation = self.evaluator.evaluate(query, chunks)
            trace.attempts.append(_attempt(current_query, chunks, evaluation))

        trace.final_verdict = evaluation.verdict
        selected = [c for c in chunks if c.chunk_id in set(evaluation.relevant_chunk_ids)] or chunks
        return CorrectiveOutcome(
            chunks=dedupe_best(selected), trace=trace, retrieval_calls=calls,
            verdict_before=verdict_before, verdict_after=evaluation.verdict,
            chunks_before=first.chunks, evaluation=evaluation,
        )


def _attempt(query: str, chunks: list[RetrievedChunk], evaluation: RetrievalEvaluation) -> RetrievalAttempt:
    return RetrievalAttempt(
        query=query, chunk_ids=[c.chunk_id for c in chunks], pages=[page_ref(c) for c in chunks], evaluation=evaluation
    )
