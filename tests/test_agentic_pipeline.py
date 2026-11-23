"""End-to-end through the HTTP API with scripted LLM stages and a fake NLI model."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.core.container import Container
from backend.main import create_app
from tests.conftest import LEAVE_PAGES, upload
from tests.helpers import FakeNLI, ScriptedLLM

CLAIM = "Every permanent employee receives 20 days of annual leave per calendar year"
ANSWER = {"answer": f"{CLAIM}. [S1]", "sufficient": True}


def make_llm(**over):
    stages = dict(
        router={"query_type": "SIMPLE", "reasoning": "single lookup"},
        evaluator={"relevant_passages": [1], "sufficient": True, "reasoning": "covers it"},
        refiner={"refined_query": "annual leave days per year"},
        answer=ANSWER,
        claims={"claims": [{"text": CLAIM, "sources": ["S1"]}], "unavailable_statements": []},
    )
    stages.update(over)
    return ScriptedLLM(**stages)


@pytest.fixture
def make_client(settings, embedder):
    clients = []

    def _make(llm, nli=None):
        c = Container.build(settings, embedder=embedder, llm=llm, nli=nli or FakeNLI())
        tc = TestClient(create_app(container=c))
        tc.__enter__()
        clients.append(tc)
        from tests.conftest import make_pdf
        assert upload(tc, ("leave.pdf", make_pdf(LEAVE_PAGES))).status_code == 200
        return tc

    yield _make
    for tc in clients:
        tc.__exit__(None, None, None)


def chat(client, q, **kw):
    r = client.post("/chat", json={"question": q, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_baseline_mode_never_calls_agentic_stages(make_client):
    llm = make_llm()
    body = chat(make_client(llm), "How many annual leave days do permanent employees get?")
    stages = {s for s, _, _ in llm.calls}
    assert stages == {"answer"} and body["status"] == "answered" and body["verification"] is None
    assert body["mode"] == "baseline" and body["trace"]["retrieval_calls"] == 1
    assert body["mechanisms"] == {"adaptive": False, "corrective": False, "verification": False}


def test_full_mode_simple_question_runs_all_three_mechanisms(make_client):
    llm = make_llm()
    body = chat(make_client(llm), "How many annual leave days do permanent employees get?", mode="full")
    assert [s for s, _, _ in llm.calls] == ["router", "evaluator", "answer", "claims"]
    assert body["trace"]["query_type"] == "SIMPLE" and body["trace"]["sub_questions"] == []
    ver = body["verification"]
    assert ver["claims"][0]["verdict"] == "SUPPORTED" and ver["nli_model"] == "fake-nli"
    assert ver["claims"][0]["supporting"]["filename"] == "leave.pdf"
    assert body["original_answer"] and body["status"] == "answered"
    assert body["citations"] and body["citations"][0]["filename"] == "leave.pdf" and body["citations"][0]["page_start"] >= 1
    assert body["trace"]["tokens"]["calls"] == 4
    assert {"routing_ms", "retrieval_ms", "generation_ms", "verification_ms", "total_ms"} <= set(body["trace"]["timings_ms"])


def test_complex_question_is_decomposed_and_each_sub_question_retrieved(make_client):
    llm = make_llm(router={"query_type": "COMPLEX", "reasoning": "two parts", "sub_questions": [
        "How many annual leave days do permanent employees receive?", "How many unused leave days can be carried forward?"]})
    body = chat(make_client(llm), "Explain annual leave entitlement and carry forward", mode="custom",
                flags={"adaptive": True, "corrective": False, "verification": False})
    units = body["trace"]["retrieval_units"]
    assert body["trace"]["query_type"] == "COMPLEX" and len(units) == 2 and body["trace"]["retrieval_calls"] == 2
    assert units[0]["query"].startswith("How many annual leave") and body["trace"]["sub_questions"][1].startswith("How many unused")
    text = " ".join(e["text"] for e in body["evidence"])
    assert "20 days" in text and "carried forward" in text  # evidence from both sub-questions was combined
    assert "evaluator" not in {s for s, _, _ in llm.calls}  # corrective was off


def test_conversational_message_is_not_answered_from_documents(make_client):
    llm = make_llm(router={"query_type": "CONVERSATIONAL", "conversational_reply": "Hi! Ask me about your documents."})
    body = chat(make_client(llm), "hello there", mode="full")
    assert body["status"] == "conversational" and body["citations"] == [] and body["evidence"] == []
    assert [s for s, _, _ in llm.calls] == ["router"]


def test_corrective_retry_is_logged_and_uses_refined_query(make_client):
    import re
    first = {"relevant_passages": [], "sufficient": False, "missing_information": "number of days", "reasoning": "wrong section"}
    seen = []

    def evaluator(user):  # first call: nothing relevant; afterwards: every passage is relevant
        seen.append(user)
        if len(seen) == 1:
            return first
        n = len(re.findall(r"^\[\d+\] \(", user, re.M))
        return {"relevant_passages": list(range(1, n + 1)), "sufficient": True, "reasoning": "now found"}

    llm = make_llm(evaluator=evaluator)
    body = chat(make_client(llm), "leave policy overview permanent employees", mode="custom",
                flags={"adaptive": False, "corrective": True, "verification": False})
    unit = body["trace"]["retrieval_units"][0]
    assert unit["verdict_before"] == "IRRELEVANT" and unit["verdict_after"] == "RELEVANT"
    corr = unit["corrective"]
    assert corr["retried"] and corr["refined_query"] == "annual leave days per year" and len(corr["attempts"]) == 2
    assert "number of days" in corr["retry_reason"] and body["trace"]["retrieval_calls"] == 2


def test_insufficient_after_retry_tells_generator_and_does_not_fabricate(make_client):
    llm = make_llm(
        evaluator={"relevant_passages": [], "sufficient": False, "missing_information": "bonus policy"},
        answer={"answer": "", "sufficient": False},
    )
    body = chat(make_client(llm), "what is the bonus policy?", mode="custom",
                flags={"adaptive": False, "corrective": True, "verification": False})
    assert body["status"] == "insufficient_information" and body["citations"] == []
    assert "insufficient" in llm.stage_calls("answer")[0] and "bonus policy" in llm.stage_calls("answer")[0]
    assert body["trace"]["retrieval_calls"] == 2  # one retry only


def test_verification_abstains_when_claims_are_unsupported(make_client):
    llm = make_llm(claims={"claims": [{"text": "Employees receive a free company car after one year", "sources": ["S1"]}]})
    body = chat(make_client(llm), "How many annual leave days do permanent employees get?", mode="full")
    assert body["status"] == "insufficient_information" and body["reason"] == "verification_abstained"
    assert body["citations"] == [] and body["original_answer"].startswith(CLAIM)
    assert body["verification"]["abstained"] and body["verification"]["claims"][0]["verdict"] == "INSUFFICIENT"


def test_verification_failure_falls_back_to_unverified_original_answer(make_client):
    def boom(_):
        return "this is not json"
    llm = make_llm(claims=boom)
    body = chat(make_client(llm), "How many annual leave days do permanent employees get?", mode="full")
    assert body["status"] == "answered" and body["answer"].startswith(CLAIM)  # baseline answer survives
    assert body["verification"]["error"] and body["verification"]["claims"] == []
    assert any(f.startswith("verification") for f in body["trace"]["fallbacks"])


def test_router_failure_falls_back_to_simple_retrieval(make_client):
    body = chat(make_client(make_llm(router="garbage")), "How many annual leave days?", mode="full")
    assert body["trace"]["router"]["error"] and body["trace"]["query_type"] == "SIMPLE" and body["status"] == "answered"


def test_invalid_mode_rejected(make_client):
    r = make_client(make_llm()).post("/chat", json={"question": "x", "mode": "turbo"})
    assert r.status_code == 422
