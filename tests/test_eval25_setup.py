"""Fast offline checks for the 25-question ablation orchestration (fake/no live LLM)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from backend.evaluation.dataset import load_dataset
from scripts.run_eval25 import _export, _load_rows


DATASET = "data/eval/questions25.provisional.json"


def test_25_gold_cases_have_expected_mix_and_unique_ids():
    qs = load_dataset(DATASET)
    assert len(qs) == 25
    assert len({q.id for q in qs}) == 25
    assert sum(q.query_type == "COMPLEX" for q in qs) == 11
    assert sum(q.query_type == "SIMPLE" for q in qs) == 14
    assert sum(q.answerability == "full" for q in qs) == 20
    assert sum(q.answerability == "partial" for q in qs) == 3
    assert sum(q.answerability == "unanswerable" for q in qs) == 2
    assert all(not q.illustrative for q in qs)


def test_question_references_only_documented_source_names():
    qs = load_dataset(DATASET)
    sources = {r.document for q in qs for r in q.relevant_evidence}
    assert sources == {
        "B_Sc_Acad_Sys_Aug23_ecddd80d28.pdf",
        "Scholarship_AY_2026_27_website_Apr_1_6c85696a82.pdf",
    }
    assert all(q.relevant_evidence for q in qs if q.answerability == "full")
    assert all(not q.relevant_evidence for q in qs if q.answerability == "unanswerable")
    assert all(q.reference_sub_questions for q in qs if q.query_type == "COMPLEX")


def test_empty_checkpoint_and_valid_saved_rows(tmp_path):
    p = tmp_path / "completed.jsonl"
    assert _load_rows(p) == []
    rows = [{"experiment": "E1", "id": "ke-01", "status": "error", "error": "offline"}]
    p.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    assert _load_rows(p) == rows
    with pytest.raises(ValueError):
        p.write_text("{broken\n", encoding="utf-8")
        _load_rows(p)


def test_export_failsafe_when_all_rows_are_errors(tmp_path):
    records = [
        {"experiment": "E1", "id": "ke-01", "status": "error",
         "gold_query_type": "SIMPLE", "error": "offline"},
    ]
    _export(tmp_path, records, ("E1", "E2", "E3", "E4", "E5"))
    per_question = pd.read_csv(tmp_path / "per_question_E1.csv")
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert len(per_question) == 1
    assert len(summary) == 1
    assert summary.loc[0, "n_errors"] == 1
    assert not (tmp_path / "per_question_E2.csv").exists()
