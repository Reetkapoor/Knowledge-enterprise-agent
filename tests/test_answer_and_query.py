import json

import pytest

from backend.schemas.chat import RetrievedChunk, Turn
from backend.services.answer_service import INSUFFICIENT_MESSAGE, AnswerGenerator
from backend.services.llm_service import LLMError
from backend.services.query_service import QueryProcessor
from tests.conftest import FakeLLM


def chunk(i, filename="Employee Handbook.pdf", ps=12, pe=12, text="Employees receive 20 days of annual leave."):
    return RetrievedChunk(
        rank=i, chunk_id=f"doc{i}:00000", document_id=f"doc{i}", filename=filename,
        page_start=ps, page_end=pe, page_label=str(ps) if ps == pe else f"{ps}-{pe}",
        text=text, similarity=0.8, distance=0.2,
    )


def reply(answer, sufficient=True):
    return lambda s, u: json.dumps({"answer": answer, "sufficient": sufficient})


def test_prompt_contains_retrieved_context_and_grounding_rules():
    llm = FakeLLM(handler=reply("20 days. [S1]"))
    AnswerGenerator(llm).generate("How many leaves?", [chunk(1), chunk(2, "Leave.pdf", 3, 4, "Carry forward 5 days.")])
    system, user = llm.calls[0]
    assert "Employees receive 20 days of annual leave." in user and "Carry forward 5 days." in user
    assert "[S1] (Employee Handbook.pdf, page 12)" in user and "[S2] (Leave.pdf, page 3-4)" in user
    assert "QUESTION: How many leaves?" in user
    assert "only facts stated in the sources" in system.lower()


def test_citations_come_from_chunk_metadata():
    llm = FakeLLM(handler=reply("You get 20 days [S1]. Carry-over is 5 days [S2]."))
    res = AnswerGenerator(llm).generate("q", [chunk(1), chunk(2, "Leave Policy.pdf", 7, 8, "Carry 5 days.")])
    assert res.status == "answered"
    assert "[Employee Handbook.pdf, p. 12]" in res.answer and "[Leave Policy.pdf, p. 7-8]" in res.answer
    assert "[S1]" not in res.answer
    c1, c2 = res.citations
    assert (c1.filename, c1.page_start, c1.page_end, c1.chunk_id) == ("Employee Handbook.pdf", 12, 12, "doc1:00000")
    assert (c2.filename, c2.page_label) == ("Leave Policy.pdf", "7-8")
    assert "20 days" in c1.passage


def test_fabricated_sources_are_dropped_and_never_cited():
    llm = FakeLLM(handler=reply("Fact A [S1]. Fact B [S9]. Fact C [Fake.pdf p.4]."))
    res = AnswerGenerator(llm).generate("q", [chunk(1)])
    assert [c.source_id for c in res.citations] == ["S1"]
    assert "S9" not in res.answer and res.trace["invalid_citation_markers"] == ["S9"]


def test_answer_with_only_invalid_citations_is_treated_as_unsupported():
    res = AnswerGenerator(FakeLLM(handler=reply("Confident claim [S5]."))).generate("q", [chunk(1)])
    assert res.status == "insufficient_information" and res.answer == INSUFFICIENT_MESSAGE
    assert res.reason == "no_valid_citations" and res.citations == []


def test_answer_without_any_citation_is_rejected():
    res = AnswerGenerator(FakeLLM(handler=reply("Employees get 25 days."))).generate("q", [chunk(1)])
    assert res.status == "insufficient_information"


def test_model_reported_insufficient():
    res = AnswerGenerator(FakeLLM(handler=reply("Not covered.", sufficient=False))).generate("q", [chunk(1)])
    assert res.status == "insufficient_information" and res.reason == "model_reported_insufficient"


def test_no_chunks_means_no_llm_call():
    llm = FakeLLM()
    res = AnswerGenerator(llm).generate("q", [])
    assert res.status == "insufficient_information" and llm.calls == []


def test_tolerates_code_fences_and_plain_text_replies():
    fenced = '```json\n{"answer": "20 days [S1]", "sufficient": true}\n```'
    assert AnswerGenerator(FakeLLM(handler=lambda s, u: fenced)).generate("q", [chunk(1)]).status == "answered"
    plain = AnswerGenerator(FakeLLM(handler=lambda s, u: "20 days [S1]")).generate("q", [chunk(1)])
    assert plain.status == "answered" and plain.citations


def test_context_budget_limits_passages():
    big = [chunk(i, text="word " * 400) for i in range(1, 6)]
    llm = FakeLLM(handler=reply("x [S1]"))
    gen = AnswerGenerator(llm, tokens_per_word=1.0, max_context_tokens=900)
    res = gen.generate("q", big)
    assert res.trace["context_chunks"] == 2


# ---- query processing ---------------------------------------------------
def test_standalone_question_skips_llm():
    llm = FakeLLM()
    pq = QueryProcessor(llm).process("What is the leave policy?", [])
    assert pq.standalone == "What is the leave policy?" and not pq.rewritten and llm.calls == []


def test_follow_up_is_rewritten_using_history():
    llm = FakeLLM(rewrite="Can annual leaves be carried forward?")
    hist = [Turn(question="What is the annual leave entitlement?", answer="20 days [Handbook.pdf, p. 12]")]
    pq = QueryProcessor(llm).process("Can they be carried forward?", hist)
    assert pq.rewritten and pq.standalone == "Can annual leaves be carried forward?"
    assert "annual leave entitlement" in llm.calls[0][1]


def test_rewrite_failure_falls_back_to_original_question():
    def boom(s, u):
        raise LLMError("timeout")

    hist = [Turn(question="a", answer="b")]
    pq = QueryProcessor(FakeLLM(handler=boom)).process("Can they be carried forward?", hist)
    assert pq.standalone == "Can they be carried forward?" and "rewrite_error" in pq.metadata


def test_rewrite_not_configured_error_propagates():
    def boom(s, u):
        raise LLMError("no key", not_configured=True)

    with pytest.raises(LLMError):
        QueryProcessor(FakeLLM(handler=boom)).process("q", [Turn(question="a", answer="b")])
