"""RAG orchestration with three independently switchable mechanisms.

    question -> [query rewrite] -> [Adaptive routing] -> retrieval -> [Corrective evaluation/retry]
             -> generation -> [claim-level Verification + correction] -> final answer

Brackets are optional. With all flags off this is the plain baseline RAG. Every agentic stage is wrapped so
that a failure falls back to the baseline behaviour for that stage (and is recorded in trace["fallbacks"]);
configuration errors (missing API key) still surface as errors.
"""
from __future__ import annotations

import logging
import time
import uuid

from backend.schemas.agentic import (
    MechanismFlags,
    QueryType,
    RetrievalVerdict,
    RoutePlan,
    VerificationResult,
)
from backend.schemas.chat import ChatResponse, RetrievedChunk, Turn
from backend.services.answer_service import (
    NO_DOCUMENTS_MESSAGE,
    AnswerGenerator,
    make_citation,
)
from backend.services.conversation_store import ConversationStore
from backend.services.corrective_rag import CorrectiveOutcome, CorrectiveRetriever
from backend.services.evidence_utils import round_robin
from backend.services.llm_service import LLMError, LLMProvider
from backend.services.query_router import QueryRouter
from backend.services.query_service import QueryProcessor
from backend.services.retrieval_service import RetrievalService
from backend.services.vector_store import VectorStore
from backend.services.verification.verifier import AnswerVerifier

logger = logging.getLogger(__name__)

_FAILED_VERDICTS = {RetrievalVerdict.IRRELEVANT, RetrievalVerdict.INSUFFICIENT_EVIDENCE}


class RAGPipeline:
    def __init__(
        self,
        query_processor: QueryProcessor,
        retriever: RetrievalService,
        generator: AnswerGenerator,
        conversations: ConversationStore,
        vector_store: VectorStore,
        router: QueryRouter | None = None,
        corrective: CorrectiveRetriever | None = None,
        verifier: AnswerVerifier | None = None,
        usage_source: LLMProvider | None = None,
    ):
        self.query_processor = query_processor
        self.retriever = retriever
        self.generator = generator
        self.conversations = conversations
        self.vector_store = vector_store
        self.router = router
        self.corrective = corrective
        self.verifier = verifier
        self.usage_source = usage_source or generator.llm

    # ------------------------------------------------------------------ public
    def ask(
        self,
        question: str,
        session_id: str | None = None,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        flags: MechanismFlags | None = None,
        mode: str | None = None,
    ) -> ChatResponse:
        flags = flags or MechanismFlags()
        mode = mode or flags.label
        session_id = session_id or uuid.uuid4().hex
        timings: dict[str, float] = {}
        fallbacks: list[str] = []
        t_all = time.perf_counter()
        usage0 = self.usage_source.usage_snapshot()

        def respond(**kw) -> ChatResponse:
            timings["total_ms"] = _ms(t_all)
            trace = kw.pop("trace", {})
            trace.update(
                mechanisms=flags.model_dump(), timings_ms=timings, fallbacks=fallbacks,
                tokens=_usage_delta(usage0, self.usage_source.usage_snapshot()),
            )
            return ChatResponse(session_id=session_id, question=question, mode=mode, mechanisms=flags, trace=trace, **kw)

        if self.vector_store.count() == 0:
            return respond(standalone_question=question, answer=NO_DOCUMENTS_MESSAGE,
                           status="insufficient_information", reason="no_documents")

        history = self.conversations.history(session_id)
        t = time.perf_counter()
        processed = self.query_processor.process(question, history)
        timings["query_ms"] = _ms(t)
        standalone = processed.standalone

        # ---- Adaptive RAG: routing -------------------------------------------------------------
        plan: RoutePlan | None = None
        if flags.adaptive and self.router:
            t = time.perf_counter()
            try:
                plan = self.router.route(standalone)
            except LLMError as exc:
                if exc.not_configured:
                    raise
                fallbacks.append(f"adaptive: {exc}")
            except Exception as exc:  # noqa: BLE001 - never let an agentic stage break the baseline path
                logger.exception("Router failed")
                fallbacks.append(f"adaptive: {type(exc).__name__}")
            timings["routing_ms"] = _ms(t)
        if plan and plan.query_type == QueryType.CONVERSATIONAL:
            reply = plan.conversational_reply or ""
            self.conversations.add(session_id, Turn(question=question, answer=reply))
            return respond(standalone_question=standalone, answer=reply, status="conversational",
                           reason="not_a_document_question", trace={"router": plan.model_dump()})

        # ---- Retrieval (+ Corrective RAG) ------------------------------------------------------
        t = time.perf_counter()
        units = plan.sub_questions if plan and plan.query_type == QueryType.COMPLEX else [standalone]
        per_unit, outcomes, retrieval_calls = [], [], 0
        for unit in units:
            chunks, outcome, calls = self._retrieve_unit(unit, top_k, document_ids, flags, fallbacks)
            per_unit.append(chunks)
            outcomes.append(outcome)
            retrieval_calls += calls
        evidence = per_unit[0] if len(per_unit) == 1 else round_robin(per_unit)
        timings["retrieval_ms"] = _ms(t)

        # ---- Generation ------------------------------------------------------------------------
        t = time.perf_counter()
        note = _evidence_note(outcomes)
        result = self.generator.generate(standalone, evidence, evidence_note=note)
        timings["generation_ms"] = _ms(t)

        original_answer = result.answer
        answer, status, reason, citations = result.answer, result.status, result.reason, list(result.citations)

        # ---- Claim-level verification ----------------------------------------------------------
        verification: VerificationResult | None = None
        if flags.verification and self.verifier and result.status == "answered":
            t = time.perf_counter()
            try:
                verification, outcome = self.verifier.verify(result)
                answer = verification.final_answer
                if outcome.abstained:
                    status, reason, citations = "insufficient_information", "verification_abstained", []
                elif not outcome.unchanged:
                    by_id = {c.chunk_id: (i, c) for i, c in enumerate(result.used_chunks, 1)}
                    citations = [make_citation(by_id[cid][1], f"S{by_id[cid][0]}") for cid in outcome.used_chunk_ids if cid in by_id]
            except LLMError as exc:
                if exc.not_configured:
                    raise
                verification = _unverified(original_answer, f"verification LLM error: {exc}")
                fallbacks.append(f"verification: {exc}")
            except Exception as exc:  # noqa: BLE001 - e.g. NLI model could not be loaded
                logger.exception("Verification failed")
                verification = _unverified(original_answer, f"{type(exc).__name__}: {str(exc)[:200]}")
                fallbacks.append(f"verification: {type(exc).__name__}")
            timings["verification_ms"] = _ms(t)

        cited_ids = {c.chunk_id for c in citations}
        for chunk in evidence:
            chunk.cited = chunk.chunk_id in cited_ids
        self.conversations.add(session_id, Turn(question=question, answer=answer))

        trace = {
            "query_rewritten": processed.rewritten,
            "router": plan.model_dump() if plan else None,
            "query_type": plan.query_type.value if plan else None,
            "sub_questions": list(plan.sub_questions) if plan else [],
            "retrieval_calls": retrieval_calls,
            "retrieval_units": [_unit_trace(u, c, o) for u, c, o in zip(units, per_unit, outcomes)],
            "evidence_note": note,
            **result.trace,
        }
        return respond(
            standalone_question=standalone, answer=answer, status=status, reason=reason, citations=citations,
            evidence=evidence, trace=trace, original_answer=original_answer if verification else None,
            verification=verification,
        )

    # ------------------------------------------------------------------ internals
    def _retrieve_unit(self, query, top_k, document_ids, flags, fallbacks):
        if flags.corrective and self.corrective:
            try:
                out = self.corrective.run(query, top_k, document_ids)
                return out.chunks, out, out.retrieval_calls
            except LLMError as exc:
                if exc.not_configured:
                    raise
                fallbacks.append(f"corrective: {exc}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Corrective retrieval failed")
                fallbacks.append(f"corrective: {type(exc).__name__}")
        res = self.retriever.retrieve(query, top_k, document_ids)
        return res.chunks, None, 1


def _evidence_note(outcomes: list[CorrectiveOutcome | None]) -> str | None:
    bad = [o for o in outcomes if o and o.verdict_after in _FAILED_VERDICTS]
    if not bad:
        return None
    missing = "; ".join(filter(None, (o.evaluation.missing_information for o in bad if o.evaluation)))
    return (
        "The retrieval check judged the available evidence insufficient"
        + (f" (missing: {missing})" if missing else "")
        + ". Answer only what the sources directly support and say clearly which part is not covered. "
        "If nothing in the sources answers the question, set sufficient to false."
    )


def _chunk_refs(chunks: list[RetrievedChunk]) -> list[dict]:
    return [{"chunk_id": c.chunk_id, "document_id": c.document_id, "filename": c.filename,
             "page_start": c.page_start, "page_end": c.page_end, "similarity": c.similarity} for c in chunks]


def _unit_trace(query: str, final: list[RetrievedChunk], o: CorrectiveOutcome | None) -> dict:
    d: dict = {"query": query, "final_chunks": _chunk_refs(final)}
    if o:
        d.update(
            corrective=o.trace.model_dump(), verdict_before=o.verdict_before.value if o.verdict_before else None,
            verdict_after=o.verdict_after.value if o.verdict_after else None, chunks_before=_chunk_refs(o.chunks_before),
        )
    return d


def _unverified(answer: str, error: str) -> VerificationResult:
    return VerificationResult(original_answer=answer, final_answer=answer, error=error)


def _usage_delta(a: dict, b: dict) -> dict:
    return {k: b.get(k, 0) - a.get(k, 0) for k in ("calls", "input_tokens", "output_tokens")}


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
