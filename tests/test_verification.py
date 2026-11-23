from __future__ import annotations

import numpy as np
import pytest

from backend.schemas.agentic import ClaimVerdict as CV
from backend.schemas.chat import RetrievedChunk
from backend.schemas.documents import DocumentRecord
from backend.services.answer_service import GenerationResult
from backend.services.verification.citations import CitationValidator
from backend.services.verification.claims import ClaimExtractor
from backend.services.verification.evidence import make_windows, missing_numbers, numbers_in, split_sentences
from backend.services.verification.nli import NLIError, label_indices, scores_from_logits
from backend.services.verification.verifier import AnswerVerifier, decide_verdict
from tests.conftest import FakeEmbedder
from tests.helpers import FakeNLI, ScriptedLLM

LEAVE = ("Annual leave entitlement. Every permanent employee receives 20 days of annual leave per calendar year. "
         "Leave is credited on the first of January.")
CARRY = ("Carry forward rules. Up to 5 unused annual leave days may be carried forward to the next year. "
         "Carried forward days expire on 31 March.")


def chunk(i, text, doc="d1", pages=(2, 2), fname="leave.pdf"):
    return RetrievedChunk(rank=i, chunk_id=f"c{i}", document_id=doc, filename=fname, page_start=pages[0], page_end=pages[1],
                          page_label=str(pages[0]), text=text, similarity=0.8, distance=0.2)


# ---------------------------------------------------------------- pure logic
def test_nli_label_mapping_is_read_from_model_config():
    idx = label_indices({0: "contradiction", 1: "entailment", 2: "neutral"})
    assert idx == {"contradiction": 0, "entailment": 1, "neutral": 2}
    idx2 = label_indices({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"})
    assert idx2["entailment"] == 0
    with pytest.raises(NLIError):
        label_indices({0: "yes", 1: "no"})


def test_scores_from_logits_are_probabilities_in_label_order():
    s = scores_from_logits(np.array([[0.0, 5.0, 0.0]]), {0: "contradiction", 1: "entailment", 2: "neutral"})[0]
    assert s.entailment > 0.98 and abs(s.entailment + s.neutral + s.contradiction - 1) < 1e-6


@pytest.mark.parametrize("e,c,expected,conflict", [
    (0.9, 0.05, CV.SUPPORTED, False), (0.1, 0.9, CV.CONTRADICTED, False), (0.3, 0.3, CV.INSUFFICIENT, False),
    (0.9, 0.7, CV.SUPPORTED, True), (0.7, 0.9, CV.CONTRADICTED, True),
])
def test_decide_verdict(e, c, expected, conflict):
    assert decide_verdict(e, c, 0.6, 0.6) == (expected, conflict)


def test_thresholds_are_configurable():
    assert decide_verdict(0.7, 0.0, 0.8, 0.8)[0] == CV.INSUFFICIENT
    assert decide_verdict(0.7, 0.0, 0.6, 0.8)[0] == CV.SUPPORTED


def test_numeric_helpers_normalise_words_and_separators():
    assert numbers_in("twenty days and 1,000 staff, 2.0 weeks") == {"20", "1000", "2"}
    assert missing_numbers("25 days of leave", "employees get twenty days") == {"25"}
    assert missing_numbers("5 days", "five days") == set()


def test_windows_cover_chunk_and_keep_provenance():
    c = chunk(1, "One. Two. Three. Four. Five. Six.")
    wins = make_windows(c, sentences=3, stride=2)
    assert wins[0].text.startswith("One") and "Six." in wins[-1].text
    assert all(w.chunk.document_id == "d1" and w.chunk.page_label == "2" for w in wins)
    assert split_sentences("no punctuation " * 50, max_words=60)  # long run-on text still splits


# ---------------------------------------------------------------- verifier
def make_verifier(claims_json, contradictions=None, registry_pages=3, numeric_guard=True):
    llm = ScriptedLLM(claims=claims_json)
    docs = {"d1": DocumentRecord(document_id="d1", filename="leave.pdf", num_pages=registry_pages)}
    existing = lambda ids: {i for i in ids if i in {"c1", "c2"}}  # noqa: E731
    v = AnswerVerifier(
        ClaimExtractor(llm), FakeNLI(contradictions), FakeEmbedder(), CitationValidator(docs.get, existing),
        entail_threshold=0.6, contradict_threshold=0.6, numeric_guard=numeric_guard,
    )
    return v


def generation(answer, marked, chunks):
    return GenerationResult(answer=answer, status="answered", used_chunks=chunks, marked_answer=marked)


def test_all_supported_keeps_original_answer_and_validates_citations():
    v = make_verifier({"claims": [
        {"text": "Every permanent employee receives 20 days of annual leave per calendar year", "sources": ["S1"]},
        {"text": "Up to 5 unused annual leave days may be carried forward to the next year", "sources": ["S2"]},
    ]})
    g = generation("ORIGINAL [leave.pdf, p. 2]", "x [S1] y [S2]", [chunk(1, LEAVE), chunk(2, CARRY)])
    res, outcome = v.verify(g)
    assert [c.verdict for c in res.claims] == [CV.SUPPORTED, CV.SUPPORTED]
    assert outcome.unchanged and res.final_answer == "ORIGINAL [leave.pdf, p. 2]"
    assert res.counts["citations_valid"] == 2 and res.counts["claims_with_valid_citation"] == 2
    assert res.claims[0].supporting.page_label == "2" and res.claims[0].supporting.filename == "leave.pdf"


def test_unsupported_claim_removed_and_answer_says_so():
    v = make_verifier({"claims": [
        {"text": "Every permanent employee receives 20 days of annual leave per calendar year", "sources": ["S1"]},
        {"text": "Employees also receive a 500 dollar travel bonus", "sources": ["S1"]},
    ]})
    res, out = v.verify(generation("a", "a [S1]", [chunk(1, LEAVE)]))
    assert [c.verdict for c in res.claims] == [CV.SUPPORTED, CV.INSUFFICIENT]
    assert "travel bonus" not in res.final_answer and "could not be verified" in res.final_answer
    assert res.claims[1].action == "removed" and res.claims[1].citation_checks[0].supports_claim is False
    assert res.counts["insufficient"] == 1 and res.counts["citations_valid"] == 1 and res.counts["citations_total"] == 2


def test_contradicted_claim_is_corrected_from_evidence_when_reliable():
    v = make_verifier(
        {"claims": [{"text": "Every permanent employee receives 25 days of annual leave per calendar year", "sources": ["S1"]}]},
        contradictions=[("20 days", "25 days")],
    )
    res, out = v.verify(generation("a", "a [S1]", [chunk(1, LEAVE)]))
    vc = res.claims[0]
    assert vc.verdict == CV.CONTRADICTED and vc.action == "corrected"
    assert "20 days of annual leave" in res.final_answer and "[leave.pdf, p. 2]" in res.final_answer
    assert "25 days" not in res.final_answer.split("Correction")[0]


def test_numeric_guard_demotes_entailment_with_wrong_number():
    claim = "Every permanent employee receives 25 days of annual leave per calendar year"
    # NLI wrongly says entailment (token overlap is high) but 25 never appears in the evidence.
    on = make_verifier({"claims": [{"text": claim, "sources": ["S1"]}]})
    res, _ = on.verify(generation("a", "a [S1]", [chunk(1, LEAVE)]))
    assert res.claims[0].verdict == CV.INSUFFICIENT and res.claims[0].demoted_by_numeric_guard
    off = make_verifier({"claims": [{"text": claim, "sources": ["S1"]}]}, numeric_guard=False)
    res, _ = off.verify(generation("a", "a [S1]", [chunk(1, LEAVE)]))
    assert res.claims[0].verdict == CV.SUPPORTED


def test_fabricated_and_invalid_citations_are_rejected_and_reattributed():
    v = make_verifier({"claims": [
        {"text": "Every permanent employee receives 20 days of annual leave per calendar year", "sources": ["S9"]}]})
    res, out = v.verify(generation("a", "a", [chunk(1, LEAVE)]))
    vc = res.claims[0]
    assert vc.verdict == CV.SUPPORTED and not vc.citation_checks[0].exists
    assert vc.reattributed and vc.final_citations == ["c1"] and not out.unchanged
    assert res.counts["citations_valid"] == 0 and res.counts["reattributed"] == 1
    assert "[leave.pdf, p. 2]" in res.final_answer  # built from chunk metadata, not model text


def test_page_outside_document_invalidates_citation():
    v = make_verifier({"claims": [{"text": "Leave is credited on the first of January", "sources": ["S1"]}]}, registry_pages=1)
    res, _ = v.verify(generation("a", "a [S1]", [chunk(1, LEAVE, pages=(2, 2))]))
    ck = res.claims[0].citation_checks[0]
    assert ck.exists and not ck.page_valid and not ck.valid and "outside document" in ck.reason


def test_cited_passage_must_itself_support_the_claim():
    # claim is true per chunk 2 but cites chunk 1 -> citation invalid, re-attributed to chunk 2
    v = make_verifier({"claims": [{"text": "Up to 5 unused annual leave days may be carried forward to the next year", "sources": ["S1"]}]})
    res, _ = v.verify(generation("a", "a [S1]", [chunk(1, LEAVE), chunk(2, CARRY)]))
    vc = res.claims[0]
    assert vc.verdict == CV.SUPPORTED and not vc.citation_checks[0].supports_claim
    assert vc.reattributed and vc.final_citations == ["c2"]


def test_abstains_when_no_claim_survives():
    v = make_verifier({"claims": [{"text": "The cafeteria serves lunch until three pm daily", "sources": ["S1"]}]})
    res, out = v.verify(generation("a", "a [S1]", [chunk(1, LEAVE)]))
    assert res.abstained and out.abstained and "could not verify" in res.final_answer


def test_conflicting_evidence_is_flagged_and_qualified():
    v = make_verifier(
        {"claims": [{"text": "Every permanent employee receives 20 days of annual leave per calendar year", "sources": ["S1"]}]},
        contradictions=[("annual leave of 18 days", "20 days")],
    )
    other = chunk(2, "Revised policy. Permanent employees get annual leave of 18 days per year.", fname="old.pdf", pages=(1, 1))
    other.document_id = "d1"
    res, out = v.verify(generation("a", "a [S1]", [chunk(1, LEAVE), other]))
    vc = res.claims[0]
    assert vc.conflict and vc.verdict == CV.SUPPORTED and vc.action == "qualified"
    assert "another passage appears to disagree" in res.final_answer


def test_unavailable_statements_are_kept_verbatim_and_not_treated_as_claims():
    ans = "Every permanent employee receives 20 days of annual leave per calendar year [S1]. The documents do not mention contractors."
    v = make_verifier({"claims": [{"text": "Every permanent employee receives 20 days of annual leave per calendar year", "sources": ["S1"]}],
                       "unavailable_statements": ["The documents do not mention contractors.", "invented text"]})
    res, _ = v.verify(generation("a", ans, [chunk(1, LEAVE)]))
    assert res.unavailable_statements == ["The documents do not mention contractors."]
