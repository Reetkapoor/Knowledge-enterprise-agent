"""Metric definitions. Every formula lives here, as a pure function, so it is documented once and unit-tested.

Notation: N(x) = number of x.

Adaptive RAG
  Query classification accuracy = N(predicted type == gold type) / N(questions)
  Decomposition recall    = N(reference sub-questions matched by a predicted one) / N(reference sub-questions)
  Decomposition precision = N(predicted sub-questions matching a reference one)   / N(predicted sub-questions)
    ("match" = cosine similarity of sentence embeddings >= threshold)
Corrective RAG (page-level gold evidence)
  Retrieval precision = N(retrieved chunks overlapping a gold page) / N(retrieved chunks)
  Evidence recall     = N(gold evidence items covered by a retrieved chunk) / N(gold evidence items)
  Evidence sufficiency = 1 if evidence recall == 1 else 0
  Correction success rate = N(retried questions whose evidence recall strictly increased) / N(retried questions)
Verification (claim level; labels are SUPPORTED / CONTRADICTED / INSUFFICIENT)
  Precision_L = TP_L / (TP_L + FP_L),  Recall_L = TP_L / (TP_L + FN_L),  F1_L = 2PR / (P + R)
  Unsupported Claim Rate = N(claims not SUPPORTED) / N(claims)
  Contradiction detection rate = Recall of the CONTRADICTED label
  Citation precision = N(valid supporting citations) / N(citations)
  Citation coverage  = N(claims with >= 1 valid citation) / N(claims)
  Abstention accuracy = N(correct answer/abstain decisions) / N(full + unanswerable questions)
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np

LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT")


def safe_div(a: float, b: float) -> float:
    return float("nan") if not b else a / b


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p, r = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    f1 = float("nan") if (np.isnan(p) or np.isnan(r) or (p + r) == 0) else 2 * p * r / (p + r)
    return p, r, f1


def accuracy(pred: Sequence, gold: Sequence) -> float:
    return safe_div(sum(p == g for p, g in zip(pred, gold)), len(gold))


# ------------------------------------------------------------------ retrieval vs gold evidence
def _overlaps(chunk: dict, gold) -> bool:
    if chunk["filename"].strip().lower() != gold.document.strip().lower():
        return False
    if not gold.pages:  # whole document is gold
        return True
    return any(chunk["page_start"] <= p <= chunk["page_end"] for p in gold.pages)


def retrieval_scores(chunks: list[dict], gold: list) -> dict[str, float]:
    """chunks: dicts with filename/page_start/page_end. gold: EvidenceRef list."""
    hit_chunks = sum(any(_overlaps(c, g) for g in gold) for c in chunks)
    covered = sum(any(_overlaps(c, g) for c in chunks) for g in gold)
    recall = safe_div(covered, len(gold))
    return {
        "precision": safe_div(hit_chunks, len(chunks)),
        "recall": recall,
        "sufficient": float(recall == 1.0) if gold else float("nan"),
    }


def correction_success(before_recall: float, after_recall: float) -> bool:
    return after_recall > before_recall


# ------------------------------------------------------------------ verification
def unsupported_claim_rate(n_supported: int, n_claims: int) -> float:
    return safe_div(n_claims - n_supported, n_claims)


def citation_precision(valid: int, total: int) -> float:
    return safe_div(valid, total)


def citation_coverage(claims_with_valid: int, total_claims: int) -> float:
    return safe_div(claims_with_valid, total_claims)


def verification_report(gold: Sequence[str], pred: Sequence[str]) -> dict:
    """Per-label and macro P/R/F1 + confusion matrix for claim verdicts."""
    out: dict = {"n": len(gold), "accuracy": accuracy(pred, gold), "per_label": {}, "confusion": {}}
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        p, r, f = prf(tp, fp, fn)
        out["per_label"][lab] = {"precision": p, "recall": r, "f1": f, "support": sum(g == lab for g in gold)}
    f1s = [v["f1"] for v in out["per_label"].values() if not np.isnan(v["f1"])]
    out["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    for g in LABELS:
        out["confusion"][g] = {p: sum(gg == g and pp == p for gg, pp in zip(gold, pred)) for p in LABELS}
    out["contradiction_detection_rate"] = out["per_label"]["CONTRADICTED"]["recall"]
    return out


# ------------------------------------------------------------------ answers
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9.%]+", " ", s.lower()).strip()


def key_fact_recall(answer: str, key_facts: Iterable[str]) -> float:
    facts = [f for f in key_facts if f.strip()]
    a = f" {_norm(answer)} "
    return safe_div(sum(f" {_norm(f)} " in a for f in facts), len(facts))


def abstention_correct(answerability: str, abstained: bool) -> float:
    """1/0 for full & unanswerable questions; NaN for partial/conflicting (no single right decision)."""
    if answerability == "unanswerable":
        return float(abstained)
    if answerability == "full":
        return float(not abstained)
    return float("nan")


def decomposition_scores(pred: list[str], ref: list[str], sim_matrix: np.ndarray, threshold: float) -> dict[str, float]:
    """sim_matrix[i, j] = cosine(pred_i, ref_j)."""
    if not ref:
        return {"recall": float("nan"), "precision": float("nan")}
    if not pred:
        return {"recall": 0.0, "precision": float("nan")}
    hit = sim_matrix >= threshold
    return {"recall": safe_div(hit.any(axis=0).sum(), len(ref)), "precision": safe_div(hit.any(axis=1).sum(), len(pred))}
