import os

import numpy as np
import pytest

from backend.schemas.documents import PageText
from backend.services.retrieval_service import RetrievalService
from backend.services.text_chunker import chunk_pages
from backend.services.vector_store import VectorStore


def index(store, embedder, doc_id, name, texts, model="fake"):
    pages = [PageText(page_number=i + 1, text=t) for i, t in enumerate(texts)]
    chunks = chunk_pages(pages, document_id=doc_id, filename=name, chunk_size=30, chunk_overlap=0, tokens_per_word=1.0)
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]), model)
    return chunks


# ---- embeddings ---------------------------------------------------------
def test_embeddings_have_one_normalized_vector_per_text(embedder):
    vecs = embedder.embed_documents(["annual leave policy", "password rotation rules", "dress code"])
    assert vecs.shape == (3, embedder.DIM)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_identical_texts_are_embedded_once_but_returned_in_order(embedder):
    texts = ["leave policy", "password rules", "leave policy", "leave policy"]
    vecs = embedder.embed_documents(texts)
    assert len(embedder.encoded_texts) == 2
    assert vecs.shape[0] == 4
    assert np.allclose(vecs[0], vecs[2]) and np.allclose(vecs[0], vecs[3])
    assert not np.allclose(vecs[0], vecs[1])


def test_query_and_documents_share_one_embedding_space(embedder):
    doc = embedder.embed_documents(["annual leave entitlement"])[0]
    q = np.array(embedder.embed_query("annual leave entitlement"))
    assert float(doc @ q) == pytest.approx(1.0, abs=1e-5)


def test_empty_batch(embedder):
    assert len(embedder.embed_documents([])) == 0


@pytest.mark.real_model
@pytest.mark.skipif(not os.getenv("RUN_REAL_MODEL_TESTS"), reason="needs model download; set RUN_REAL_MODEL_TESTS=1")
def test_real_sentence_transformer_model():
    from backend.services.embedding_service import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", 8, "Represent this sentence for searching relevant passages: ")
    docs = emb.embed_documents(["Employees get 20 days of annual leave.", "Passwords rotate every 90 days."])
    q = np.array(emb.embed_query("How many vacation days do I get?"))
    assert docs.shape == (2, 384)
    assert float(docs[0] @ q) > float(docs[1] @ q)


# ---- vector store -------------------------------------------------------
def test_insert_and_semantic_retrieval(settings, embedder):
    store = VectorStore(settings.chroma_dir)
    index(store, embedder, "d1", "leave.pdf", ["Annual leave is 20 days per year."])
    index(store, embedder, "d2", "it.pdf", ["Passwords must be rotated every 90 days."])
    hits = store.query(embedder.embed_query("how many days of annual leave"), top_k=2)
    assert hits[0].filename == "leave.pdf"
    assert hits[0].similarity > hits[1].similarity
    assert hits[0].page_label == "1" and hits[0].distance == pytest.approx(1 - hits[0].similarity, abs=1e-3)


def test_filter_by_document_id(settings, embedder):
    store = VectorStore(settings.chroma_dir)
    index(store, embedder, "d1", "leave.pdf", ["Annual leave is 20 days per year."])
    index(store, embedder, "d2", "it.pdf", ["Passwords must be rotated every 90 days."])
    q = embedder.embed_query("annual leave days")
    assert {h.document_id for h in store.query(q, 5, ["d2"])} == {"d2"}
    assert {h.document_id for h in store.query(q, 5, ["d1", "d2"])} == {"d1", "d2"}


def test_readding_same_chunks_does_not_duplicate(settings, embedder):
    store = VectorStore(settings.chroma_dir)
    index(store, embedder, "d1", "leave.pdf", ["Annual leave is 20 days per year."])
    n = store.count()
    index(store, embedder, "d1", "leave.pdf", ["Annual leave is 20 days per year."])
    assert store.count() == n


def test_delete_document_removes_all_its_chunks(settings, embedder):
    store = VectorStore(settings.chroma_dir)
    chunks = index(store, embedder, "d1", "leave.pdf", ["Annual leave is 20 days per year. " * 10])
    index(store, embedder, "d2", "it.pdf", ["Passwords must be rotated every 90 days."])
    assert store.delete_document("d1") == len(chunks)
    assert all(h.document_id != "d1" for h in store.query(embedder.embed_query("annual leave"), 10))
    assert store.delete_document("d1") == 0


def test_store_survives_restart(settings, embedder):
    index(VectorStore(settings.chroma_dir), embedder, "d1", "leave.pdf", ["Annual leave is 20 days per year."])
    reopened = VectorStore(settings.chroma_dir)
    assert reopened.count() == 1
    assert reopened.query(embedder.embed_query("annual leave"), 1)[0].filename == "leave.pdf"


def test_empty_store_returns_no_results(settings, embedder):
    assert VectorStore(settings.chroma_dir).query(embedder.embed_query("anything"), 3) == []


# ---- retrieval ----------------------------------------------------------
def test_retrieval_with_no_matching_evidence_returns_nothing(settings, embedder):
    store = VectorStore(settings.chroma_dir)
    index(store, embedder, "d1", "leave.pdf", ["Annual leave is 20 days per year."])
    retriever = RetrievalService(embedder, store, top_k=3, min_similarity=0.2)
    result = retriever.retrieve("What is the capital of France?")
    assert result.chunks == []
    assert result.candidates_considered == 1 and result.dropped_below_threshold == 1


def test_retrieval_ranks_and_limits_top_k(settings, embedder):
    store = VectorStore(settings.chroma_dir)
    for i in range(5):
        index(store, embedder, f"d{i}", f"f{i}.pdf", [f"annual leave topic number {i} days"])
    result = RetrievalService(embedder, store, top_k=2, min_similarity=0.0).retrieve("annual leave days")
    assert len(result.chunks) == 2 and [c.rank for c in result.chunks] == [1, 2]
    assert RetrievalService(embedder, store, 2, 0.0).retrieve("annual leave days", top_k=4).chunks.__len__() == 4
