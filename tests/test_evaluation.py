from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient  # noqa: F401

from backend.core.container import Container
from backend.evaluation import metrics as M
from backend.evaluation.dataset import EvalQuestion, EvidenceRef, load_dataset, load_labelled_claims
from backend.evaluation.runner import EXPERIMENTS, ExperimentRunner, summarize
from tests.conftest import LEAVE_PAGES, make_pdf
from tests.helpers import FakeNLI
from tests.test_agentic_pipeline import ANSWER, make_llm


# ---------------------------------------------------------------- formulas
def test_prf_and_f1():
    p, r, f = M.prf(tp=6, fp=2, fn=4)
    assert (p, r) == (0.75, 0.6) and f == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert math.isnan(M.prf(0, 0, 0)[2])


def test_claim_and_citation_formulas():
    assert M.unsupported_claim_rate(n_supported=7, n_claims=10) == pytest.approx(0.3)
    assert M.citation_precision(valid=3, total=4) == 0.75
    assert M.citation_coverage(claims_with_valid=5, total_claims=10) == 0.5
    assert math.isnan(M.citation_precision(0, 0))


def test_verification_report_per_label_and_confusion():
    gold = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "INSUFFICIENT"]
    pred = ["SUPPORTED", "INSUFFICIENT", "CONTRADICTED", "INSUFFICIENT", "SUPPORTED"]
    rep = M.verification_report(gold, pred)
    s = rep["per_label"]["SUPPORTED"]
    assert (s["precision"], s["recall"]) == (0.5, 0.5) and rep["contradiction_detection_rate"] == 1.0
    assert rep["confusion"]["SUPPORTED"] == {"SUPPORTED": 1, "CONTRADICTED": 0, "INSUFFICIENT": 1}
    assert rep["accuracy"] == 0.6


def test_retrieval_scores_use_page_overlap_and_filename():
    gold = [EvidenceRef(document="Leave.pdf", pages=[2]), EvidenceRef(document="leave.pdf", pages=[9])]
    chunks = [{"filename": "leave.pdf", "page_start": 1, "page_end": 2}, {"filename": "other.pdf", "page_start": 9, "page_end": 9}]
    s = M.retrieval_scores(chunks, gold)
    assert s == {"precision": 0.5, "recall": 0.5, "sufficient": 0.0}
    assert M.correction_success(0.5, 1.0) and not M.correction_success(0.5, 0.5)


def test_abstention_and_key_facts_and_decomposition():
    assert M.abstention_correct("unanswerable", True) == 1.0 and M.abstention_correct("full", True) == 0.0
    assert math.isnan(M.abstention_correct("partial", True))
    assert M.key_fact_recall("You get 20 days per year.", ["20 days", "5 days"]) == 0.5
    sim = np.array([[0.9, 0.1], [0.2, 0.3]])  # pred x ref
    d = M.decomposition_scores(["a", "b"], ["x", "y"], sim, 0.75)
    assert d == {"recall": 0.5, "precision": 0.5}


# ---------------------------------------------------------------- dataset loading
def test_loads_json_and_skips_illustrative(tmp_path):
    rows = [
        {"id": "a", "question": "q", "answerability": "full", "query_type": "SIMPLE", "illustrative": True},
        {"id": "b", "question": "q2", "answerability": "unanswerable", "query_type": "SIMPLE"},
    ]
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"questions": rows}))
    assert [q.id for q in load_dataset(p)] == ["b"]
    assert len(load_dataset(p, include_illustrative=True)) == 2


def test_loads_csv_with_json_columns(tmp_path):
    df = pd.DataFrame([{"id": "a", "question": "q", "answerability": "full", "query_type": "COMPLEX", "key_facts": '["20 days"]',
                        "relevant_evidence": '[{"document": "x.pdf", "pages": [1]}]', "reference_sub_questions": "", "illustrative": "false"}])
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    q = load_dataset(p)[0]
    assert q.key_facts == ["20 days"] and q.relevant_evidence[0].pages == [1] and q.reference_sub_questions == []


def test_duplicate_ids_rejected(tmp_path):
    r = {"id": "a", "question": "q", "answerability": "full", "query_type": "SIMPLE"}
    p = tmp_path / "d.json"
    p.write_text(json.dumps([r, r]))
    with pytest.raises(ValueError):
        load_dataset(p)


def test_labelled_claims_skip_unlabelled_rows(tmp_path):
    df = pd.DataFrame([
        {"claim": "c1", "evidence": "p1 || p2", "system_label": "SUPPORTED", "human_label": "SUPPORTED"},
        {"claim": "c2", "evidence": "p3", "system_label": "INSUFFICIENT", "human_label": ""},
    ])
    p = tmp_path / "c.csv"
    df.to_csv(p, index=False)
    cs = load_labelled_claims(p)
    assert len(cs) == 1 and cs[0].evidence == ["p1", "p2"] and cs[0].system_label == "SUPPORTED"


def test_shipped_samples_are_illustrative_and_parse():
    assert load_dataset("data/eval/questions.sample.json") == []
    assert len(load_dataset("data/eval/questions.sample.json", include_illustrative=True)) == 5
    assert len(load_labelled_claims("data/eval/labelled_claims.sample.jsonl", include_illustrative=True)) == 3
    assert load_labelled_claims("data/eval/labelled_claims.sample.jsonl") == []


# ---------------------------------------------------------------- runner end to end (fake models)
def _answer(user: str):
    if "sabbatical" in user:
        return {"answer": "", "sufficient": False}
    return ANSWER


def _router(user: str):
    if "carry forward" in user and "Explain" in user:
        return {"query_type": "COMPLEX", "reasoning": "two parts", "sub_questions": [
            "How many annual leave days do permanent employees receive?", "How many unused leave days can be carried forward?"]}
    return {"query_type": "SIMPLE", "reasoning": "single lookup"}


@pytest.fixture
def env(settings, embedder):
    llm = make_llm(answer=_answer, router=_router)
    c = Container.build(settings, embedder=embedder, llm=llm, nli=FakeNLI())
    assert c.documents.ingest("leave.pdf", make_pdf(LEAVE_PAGES)).outcome == "indexed"
    return c


QUESTIONS = [
    EvalQuestion(id="q1", question="How many annual leave days do permanent employees get?", answerability="full", query_type="SIMPLE",
                 key_facts=["20 days"], relevant_evidence=[EvidenceRef(document="leave.pdf", pages=[2])],
                 expected_citations=[EvidenceRef(document="leave.pdf", pages=[2])]),
    EvalQuestion(id="q2", question="Explain annual leave and carry forward rules", answerability="full", query_type="COMPLEX",
                 reference_sub_questions=["How many annual leave days do permanent employees receive?", "How many unused leave days can be carried forward?"],
                 relevant_evidence=[EvidenceRef(document="leave.pdf", pages=[2]), EvidenceRef(document="leave.pdf", pages=[3])]),
    EvalQuestion(id="q3", question="What is the sabbatical policy?", answerability="unanswerable", query_type="SIMPLE"),
]


def test_runner_produces_all_experiment_summaries(env):
    runner = ExperimentRunner(env.pipeline, env.embedder, verifier=env.pipeline.verifier, decomposition_threshold=0.75)
    summaries = {}
    for exp in EXPERIMENTS:
        df = runner.run(exp, QUESTIONS)
        assert len(df) == 3 and (df.status != "error").all(), df[["id", "status", "error"]]
        summaries[exp] = summarize(df, exp)

    e1, e2, e5 = summaries["E1"], summaries["E2"], summaries["E5"]
    assert math.isnan(e1["query_type_accuracy"]) and e2["query_type_accuracy"] == 1.0  # router only runs with Adaptive
    assert e2["decomposition_recall"] == 1.0 and e2["retrieval_ops_complex"] == 2 and e2["retrieval_ops_simple"] == 1
    assert e1["retrieval_ops_per_query"] == 1
    # unanswerable question: model says insufficient -> abstention counted correct; answerable ones answered
    assert e1["abstention_accuracy"] == 1.0 and e1["false_answer_rate_unanswerable"] == 0.0
    # same instrument on every experiment: E1 answers were measured post hoc, E5 in the pipeline
    assert e1["claims_total"] > 0 and e5["claims_total"] > 0 and e1["unsupported_claim_rate"] == 0.0
    # the scripted generator always cites [S1]; when S1 is the wrong passage the validator rightly rejects it
    assert 0 < e1["citation_precision"] <= 1.0 and 0 < e5["citation_coverage"] <= 1.0
    assert e5["avg_llm_calls"] > e1["avg_llm_calls"] and e1["key_fact_recall"] == 1.0
    assert e5["evidence_recall_before"] <= e5["evidence_recall_after"]


def test_runner_survives_a_failing_question(env, monkeypatch):
    runner = ExperimentRunner(env.pipeline, env.embedder)
    real = env.pipeline.ask

    def flaky(question, *a, **k):
        if "sabbatical" in question:
            raise RuntimeError("boom")
        return real(question, *a, **k)

    monkeypatch.setattr(env.pipeline, "ask", flaky)
    df = runner.run("E1", QUESTIONS)
    assert list(df.status).count("error") == 1 and summarize(df, "E1")["n_errors"] == 1
