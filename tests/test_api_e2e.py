"""API-level tests, including the full upload -> answer workflow with real PDFs, real chunking,
real ChromaDB persistence; only the embedding model and LLM are replaced by test doubles."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from backend.core.container import Container
from backend.main import create_app
from backend.services.llm_service import AnthropicProvider, LLMError
from tests.conftest import FakeEmbedder, FakeLLM, upload


def ids_by_name(resp):
    return {r["filename"]: r["document_id"] for r in resp.json()["results"]}


def test_end_to_end_upload_to_cited_answer(client, leave_pdf, it_pdf, handbook_pdf, llm):
    resp = upload(client, ("Leave Policy.pdf", leave_pdf), ("IT Security Policy.pdf", it_pdf),
                  ("Employee Handbook.pdf", handbook_pdf))
    assert resp.status_code == 200
    results = {r["filename"]: r for r in resp.json()["results"]}
    assert all(r["status"] == "indexed" for r in results.values())
    assert results["Leave Policy.pdf"]["num_pages"] == 3 and results["Leave Policy.pdf"]["num_chunks"] >= 1

    docs = client.get("/documents").json()
    assert docs["total"] == 3 and {d["status"] for d in docs["documents"]} == {"indexed"}

    r = client.post("/chat", json={"question": "What is the annual leave entitlement?"}).json()
    assert r["status"] == "answered"
    top = r["citations"][0]
    assert top["filename"] == "Leave Policy.pdf"
    assert top["page_start"] <= 2 <= top["page_end"]                       # page 2 holds the entitlement
    assert f"[Leave Policy.pdf, p. {top['page_label']}]" in r["answer"]
    assert "20 days" in r["evidence"][0]["text"] and r["evidence"][0]["similarity"] > 0.2
    assert r["evidence"][0]["cited"] is True
    assert "20 days" in llm.answer_calls[0][1]                             # retrieved text reached the LLM


def test_cited_pages_exist_in_the_cited_document(client, leave_pdf, it_pdf):
    upload(client, ("Leave Policy.pdf", leave_pdf), ("IT Security Policy.pdf", it_pdf))
    pages = {"Leave Policy.pdf": 3, "IT Security Policy.pdf": 2}
    for q in ["How many unused leave days carry forward?", "How often are passwords rotated?"]:
        r = client.post("/chat", json={"question": q}).json()
        for c in r["citations"]:
            assert 1 <= c["page_start"] <= c["page_end"] <= pages[c["filename"]]
            assert c["chunk_id"] in {e["chunk_id"] for e in r["evidence"]}
    r = client.post("/chat", json={"question": "How often are passwords rotated?"}).json()
    assert r["citations"][0]["filename"] == "IT Security Policy.pdf"


def test_question_not_in_documents_gets_insufficient_message_without_llm_call(client, leave_pdf, llm):
    upload(client, ("Leave Policy.pdf", leave_pdf))
    r = client.post("/chat", json={"question": "What is the capital of France?"}).json()
    assert r["status"] == "insufficient_information" and r["citations"] == [] and r["evidence"] == []
    assert "couldn't find" in r["answer"] and llm.calls == []


def test_chat_before_any_upload(client):
    r = client.post("/chat", json={"question": "anything?"}).json()
    assert r["status"] == "insufficient_information" and r["reason"] == "no_documents"


def test_follow_up_question_uses_history_but_retrieves_evidence(client, leave_pdf, llm):
    upload(client, ("Leave Policy.pdf", leave_pdf))
    first = client.post("/chat", json={"question": "What is the annual leave entitlement?", "session_id": "s1"}).json()
    llm.rewrite = "Can unused annual leave days be carried forward?"
    second = client.post("/chat", json={"question": "Can they be carried forward?", "session_id": "s1"}).json()
    assert second["standalone_question"] == llm.rewrite and second["trace"]["query_rewritten"]
    assert second["status"] == "answered"
    assert any("carried forward" in e["text"].lower() for e in second["evidence"])
    assert second["session_id"] == first["session_id"] == "s1"
    # a different session has no history, so no rewrite call is made for it
    before = len(llm.calls)
    client.post("/chat", json={"question": "Can they be carried forward?", "session_id": "other"})
    assert not any("standalone search question" in c[0] for c in llm.calls[before:])
    assert client.delete("/chat/s1").json() == {"cleared": True}


def test_duplicate_upload_is_not_reindexed(client, leave_pdf, container):
    upload(client, ("Leave Policy.pdf", leave_pdf))
    n = container.vector_store.count()
    r = upload(client, ("Copy of Leave.pdf", leave_pdf)).json()["results"][0]
    assert r["status"] == "duplicate" and container.vector_store.count() == n
    assert client.get("/documents").json()["total"] == 1


def test_bad_files_are_reported_and_do_not_block_good_ones(client, leave_pdf):
    files = [
        ("files", ("notes.txt", b"hello", "text/plain")),
        ("files", ("broken.pdf", b"%PDF-1.4 not really", "application/pdf")),
        ("files", ("Leave Policy.pdf", leave_pdf, "application/pdf")),
    ]
    results = {r["filename"]: r for r in client.post("/documents/upload", files=files).json()["results"]}
    assert results["notes.txt"]["status"] == "rejected"
    assert results["broken.pdf"]["status"] == "failed" and "opened" in results["broken.pdf"]["message"]
    assert results["Leave Policy.pdf"]["status"] == "indexed"
    docs = {d["filename"]: d for d in client.get("/documents").json()["documents"]}
    assert docs["broken.pdf"]["status"] == "failed" and docs["broken.pdf"]["error"]   # failure is visible, not hidden
    assert client.get("/health").json()["failed_documents"] == 1


def test_oversized_upload_rejected(client):
    big = b"%PDF-1.4" + b"0" * (1024 * 1024 + 10)  # limit is 1 MB in test settings
    r = upload(client, ("big.pdf", big)).json()["results"][0]
    assert r["status"] == "rejected" and "limit" in r["message"]


def test_delete_removes_document_chunks_and_files(client, leave_pdf, it_pdf, container, settings):
    ids = ids_by_name(upload(client, ("Leave Policy.pdf", leave_pdf), ("IT Security Policy.pdf", it_pdf)))
    leave_id = ids["Leave Policy.pdf"]
    assert (settings.upload_dir / f"{leave_id}.pdf").exists()
    resp = client.delete(f"/documents/{leave_id}")
    assert resp.status_code == 200 and resp.json()["chunks_removed"] >= 1
    assert not (settings.upload_dir / f"{leave_id}.pdf").exists()
    assert client.get("/documents").json()["total"] == 1
    assert all(c.document_id != leave_id for c in container.vector_store.query(
        container.embedder.embed_query("annual leave"), 10))
    r = client.post("/chat", json={"question": "What is the annual leave entitlement?"}).json()
    assert r["status"] == "insufficient_information"
    assert client.delete(f"/documents/{leave_id}").status_code == 404


def test_restricting_chat_to_selected_documents(client, leave_pdf, it_pdf):
    ids = ids_by_name(upload(client, ("Leave Policy.pdf", leave_pdf), ("IT Security Policy.pdf", it_pdf)))
    r = client.post("/chat", json={"question": "annual leave days", "document_ids": [ids["IT Security Policy.pdf"]]}).json()
    assert all(e["filename"] == "IT Security Policy.pdf" for e in r["evidence"])


def test_index_and_registry_persist_across_restart(settings, leave_pdf, llm):
    c1 = Container.build(settings, embedder=FakeEmbedder(), llm=llm)
    with TestClient(create_app(container=c1)) as client:
        upload(client, ("Leave Policy.pdf", leave_pdf))
    c2 = Container.build(settings, embedder=FakeEmbedder(), llm=FakeLLM())   # simulates a restart
    with TestClient(create_app(container=c2)) as client:
        assert client.get("/documents").json()["total"] == 1
        assert client.get("/health").json()["total_chunks"] >= 1
        r = client.post("/chat", json={"question": "What is the annual leave entitlement?"}).json()
        assert r["status"] == "answered" and r["citations"][0]["filename"] == "Leave Policy.pdf"


def test_interrupted_processing_is_reported_after_restart(settings, llm):
    settings.registry_path.write_text(json.dumps([{
        "document_id": "abc", "filename": "x.pdf", "status": "processing",
        "upload_timestamp": "2026-01-01T00:00:00Z"}]))
    c = Container.build(settings, embedder=FakeEmbedder(), llm=llm)
    rec = c.registry.get("abc")
    assert rec.status.value == "failed" and "interrupted" in rec.error


def test_validation_errors(client):
    assert client.post("/chat", json={"question": "   "}).status_code == 422
    assert client.post("/chat", json={"question": "x" * 2500}).status_code == 422
    assert client.post("/chat", json={"question": "ok", "top_k": 0}).status_code == 422
    assert client.post("/documents/upload").status_code == 422


def test_llm_failure_returns_502_not_a_crash(settings, leave_pdf):
    def boom(s, u):
        raise LLMError("LLM service unavailable after retries (HTTP 503).")

    c = Container.build(settings, embedder=FakeEmbedder(), llm=FakeLLM(handler=boom))
    with TestClient(create_app(container=c)) as client:
        upload(client, ("Leave Policy.pdf", leave_pdf))
        r = client.post("/chat", json={"question": "What is the annual leave entitlement?"})
        assert r.status_code == 502 and "unavailable" in r.json()["detail"]
        assert client.get("/health").status_code == 200       # service still up


def test_missing_api_key_gives_503_and_never_leaks_secrets(settings, leave_pdf):
    settings.llm_api_key = type(settings.llm_api_key)("")
    c = Container.build(settings, embedder=FakeEmbedder(), llm=AnthropicProvider(settings))
    with TestClient(create_app(container=c)) as client:
        upload(client, ("Leave Policy.pdf", leave_pdf))
        r = client.post("/chat", json={"question": "What is the annual leave entitlement?"})
        assert r.status_code == 503 and "LLM_API_KEY" in r.json()["detail"]
        assert client.get("/health").json()["llm_configured"] is False


def test_health_never_exposes_api_key(client):
    body = client.get("/health").text
    assert "test-key" not in body
    h = client.get("/health").json()
    assert h["vector_db"] == "ok" and h["embedding_model"] == "fake-hash-embedder"
