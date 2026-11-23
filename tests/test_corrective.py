from __future__ import annotations

from backend.schemas.agentic import RetrievalVerdict as V
from backend.schemas.chat import RetrievedChunk
from backend.services.corrective_rag import (
    CorrectiveRetriever, QueryRefiner, RetrievalEvaluator, derive_verdict,
)
from backend.schemas.chat import RetrievalResult
from tests.helpers import ScriptedLLM


def chunk(i: int, text="text", sim=0.8) -> RetrievedChunk:
    return RetrievedChunk(rank=i, chunk_id=f"c{i}", document_id="d1", filename="hr.pdf", page_start=i, page_end=i,
                          page_label=str(i), text=text, similarity=sim, distance=1 - sim)


def test_verdict_rules():
    assert derive_verdict(0, 0, False, 0.5) == V.IRRELEVANT
    assert derive_verdict(5, 0, True, 0.5) == V.IRRELEVANT
    assert derive_verdict(5, 2, False, 0.5) == V.INSUFFICIENT_EVIDENCE
    assert derive_verdict(5, 2, True, 0.5) == V.PARTIALLY_RELEVANT  # sufficient but noisy
    assert derive_verdict(4, 3, True, 0.5) == V.RELEVANT


class StubRetriever:
    """Returns different chunks depending on the query so we can see the retry change the evidence."""

    def __init__(self, by_query: dict[str, list[RetrievedChunk]]):
        self.by_query, self.queries = by_query, []

    def retrieve(self, query, top_k=None, document_ids=None):
        self.queries.append(query)
        return RetrievalResult(query=query, chunks=[c.model_copy() for c in self.by_query.get(query, [])])


def build(llm, retriever, max_retries=1):
    return CorrectiveRetriever(retriever, RetrievalEvaluator(llm), QueryRefiner(llm), max_retries=max_retries)


def test_no_retry_when_relevant_and_sufficient():
    llm = ScriptedLLM(evaluator={"relevant_passages": [1], "sufficient": True, "reasoning": "ok"})
    r = StubRetriever({"q": [chunk(1)]})
    out = build(llm, r).run("q")
    assert r.queries == ["q"] and not out.trace.retried and out.retrieval_calls == 1
    assert out.verdict_before == out.verdict_after == V.RELEVANT


def test_retry_once_with_refined_query_and_success():
    verdicts = iter([
        {"relevant_passages": [], "sufficient": False, "missing_information": "carry forward limit", "reasoning": "off topic"},
        {"relevant_passages": [1, 2], "sufficient": True, "reasoning": "found it"},
    ])
    llm = ScriptedLLM(evaluator=lambda _: next(verdicts), refiner={"refined_query": "annual leave carry forward days"})
    r = StubRetriever({"leave rules": [chunk(9)], "annual leave carry forward days": [chunk(1), chunk(2)]})
    out = build(llm, r).run("leave rules")
    assert r.queries == ["leave rules", "annual leave carry forward days"]
    assert out.trace.retried and out.trace.refined_query == "annual leave carry forward days"
    assert "IRRELEVANT" in out.trace.retry_reason and "carry forward limit" in out.trace.retry_reason
    assert (out.verdict_before, out.verdict_after) == (V.IRRELEVANT, V.RELEVANT)
    assert {c.chunk_id for c in out.chunks} == {"c1", "c2"} and out.retrieval_calls == 2
    assert len(out.trace.attempts) == 2


def test_retry_limited_to_one_and_insufficient_evidence_is_passed_on():
    llm = ScriptedLLM(
        evaluator={"relevant_passages": [1], "sufficient": False, "missing_information": "x"},
        refiner={"refined_query": "better words"},
    )
    r = StubRetriever({"q": [chunk(1)], "better words": [chunk(2)]})
    out = build(llm, r, max_retries=1).run("q")
    assert r.queries == ["q", "better words"]  # exactly one retry, never a third search
    assert out.verdict_after == V.INSUFFICIENT_EVIDENCE and out.chunks  # evidence still forwarded, not fabricated


def test_retry_count_is_configurable_and_zero_disables():
    llm = ScriptedLLM(evaluator={"relevant_passages": [], "sufficient": False}, refiner={"refined_query": "zzz"})
    r = StubRetriever({"q": [chunk(1)]})
    out = build(llm, r, max_retries=0).run("q")
    assert r.queries == ["q"] and not out.trace.retried


def test_refiner_falls_back_when_llm_repeats_failed_query():
    llm = ScriptedLLM(refiner={"refined_query": "q"})
    from backend.schemas.agentic import RetrievalEvaluation
    refined = QueryRefiner(llm).refine("q", "q", RetrievalEvaluation(verdict=V.IRRELEVANT, missing_information="notice period"), [])
    assert refined != "q" and "notice period" in refined


def test_evaluator_failure_passes_retrieval_through_and_flags_error():
    llm = ScriptedLLM(evaluator="not json at all")
    r = StubRetriever({"q": [chunk(1)]})
    out = build(llm, r).run("q")
    assert out.evaluation.error and not out.trace.retried and out.chunks
