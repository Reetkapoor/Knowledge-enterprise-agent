"""Runs experiments E1-E5 over a labelled dataset and aggregates the metrics defined in metrics.py."""
from __future__ import annotations

import logging
import math
import time
import uuid
from typing import Callable

import numpy as np
import pandas as pd

from backend.evaluation import metrics as M
from backend.evaluation.dataset import EvalQuestion
from backend.evaluation.judge import judge_answer
from backend.schemas.agentic import MechanismFlags
from backend.schemas.chat import ChatResponse
from backend.services.answer_service import GenerationResult
from backend.services.embedding_service import BaseEmbedder
from backend.services.llm_service import LLMProvider
from backend.services.rag_pipeline import RAGPipeline
from backend.services.verification.verifier import AnswerVerifier

logger = logging.getLogger(__name__)

EXPERIMENTS: dict[str, MechanismFlags] = {
    "E1": MechanismFlags(),
    "E2": MechanismFlags(adaptive=True),
    "E3": MechanismFlags(corrective=True),
    "E4": MechanismFlags(verification=True),
    "E5": MechanismFlags(adaptive=True, corrective=True, verification=True),
}
EXPERIMENT_NAMES = {
    "E1": "Baseline RAG", "E2": "Baseline + Adaptive", "E3": "Baseline + Corrective",
    "E4": "Baseline + Verification", "E5": "Full system",
}
NAN = float("nan")


class ExperimentRunner:
    def __init__(
        self, pipeline: RAGPipeline, embedder: BaseEmbedder, verifier: AnswerVerifier | None = None,
        judge_llm: LLMProvider | None = None, decomposition_threshold: float = 0.75, measure_draft: bool = True,
    ):
        self.pipeline, self.embedder, self.verifier = pipeline, embedder, verifier
        self.judge_llm, self.decomposition_threshold = judge_llm, decomposition_threshold
        self.measure_draft = measure_draft  # post-hoc claim verification of E1-E3 answers (measurement only)

    def run(self, exp_id: str, questions: list[EvalQuestion], top_k: int | None = None,
            document_ids: list[str] | None = None, on_progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
        flags = EXPERIMENTS[exp_id]
        rows = []
        for i, q in enumerate(questions, 1):
            rows.append(self._run_one(exp_id, flags, q, top_k, document_ids))
            if on_progress:
                on_progress(i, len(questions))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _run_one(self, exp_id, flags, q: EvalQuestion, top_k, document_ids) -> dict:
        row: dict = {"experiment": exp_id, "id": q.id, "question": q.question, "answerability": q.answerability,
                     "gold_query_type": q.query_type, "error": ""}
        t0 = time.perf_counter()
        try:
            resp = self.pipeline.ask(q.question, uuid.uuid4().hex, top_k, document_ids, flags, flags.label)
        except Exception as exc:  # noqa: BLE001 - one bad question must not kill a 200-question run
            logger.exception("Question %s failed", q.id)
            row.update(error=f"{type(exc).__name__}: {str(exc)[:200]}", status="error", wall_ms=_ms(t0))
            return row
        row["wall_ms"] = _ms(t0)
        row.update(self._basic(resp, q))
        row.update(self._retrieval(resp, q))
        row.update(self._decomposition(resp, q))
        row.update(self._verification(resp, q))
        row.update(self._answer_quality(resp, q))
        return row

    def _basic(self, resp: ChatResponse, q: EvalQuestion) -> dict:
        t = resp.trace
        abstained = resp.status == "insufficient_information"
        return {
            "status": resp.status, "reason": resp.reason or "", "abstained": abstained, "answer": resp.answer,
            "original_answer": resp.original_answer or resp.answer,
            "pred_query_type": t.get("query_type") or "",
            "latency_ms": t["timings_ms"]["total_ms"], "retrieval_ms": t["timings_ms"].get("retrieval_ms", NAN),
            "retrieval_calls": t.get("retrieval_calls", NAN), "n_units": len(t.get("retrieval_units", [])) or 1,
            "tokens_in": t["tokens"]["input_tokens"], "tokens_out": t["tokens"]["output_tokens"], "llm_calls": t["tokens"]["calls"],
            "fallbacks": "; ".join(t.get("fallbacks", [])),
            "abstention_correct": M.abstention_correct(q.answerability, abstained),
        }

    def _retrieval(self, resp: ChatResponse, q: EvalQuestion) -> dict:
        units = resp.trace.get("retrieval_units", [])
        after = [c for u in units for c in u["final_chunks"]]
        before = [c for u in units for c in (u.get("chunks_before") if "chunks_before" in u else u["final_chunks"])]
        retried = any(u.get("corrective", {}).get("retried") for u in units)
        out = {"correction_retried": retried, "attempts_per_unit": np.mean([len(u["corrective"]["attempts"]) if "corrective" in u else 1 for u in units]) if units else NAN}
        if q.relevant_evidence and units:
            b, a = M.retrieval_scores(before, q.relevant_evidence), M.retrieval_scores(after, q.relevant_evidence)
            out.update(
                retr_precision_before=b["precision"], retr_precision_after=a["precision"],
                evidence_recall_before=b["recall"], evidence_recall_after=a["recall"],
                sufficient_before=b["sufficient"], sufficient_after=a["sufficient"],
                correction_success=float(M.correction_success(b["recall"], a["recall"])) if retried else NAN,
            )
        verdicts = [u.get("verdict_before") for u in units if u.get("verdict_before")]
        out["system_verdict_before"] = verdicts[0] if verdicts else ""
        out["system_verdict_after"] = next((u["verdict_after"] for u in units if u.get("verdict_after")), "")
        return out

    def _decomposition(self, resp: ChatResponse, q: EvalQuestion) -> dict:
        pred = resp.trace.get("sub_questions") or []
        out = {"n_sub_questions": len(pred), "decomp_recall": NAN, "decomp_precision": NAN}
        if resp.trace.get("router") is not None and q.reference_sub_questions:
            if pred:
                sim = self.embedder.embed_documents(pred) @ self.embedder.embed_documents(q.reference_sub_questions).T
                s = M.decomposition_scores(pred, q.reference_sub_questions, sim, self.decomposition_threshold)
            else:
                s = M.decomposition_scores([], q.reference_sub_questions, np.zeros((0, 0)), 1.0)
            out.update(decomp_recall=s["recall"], decomp_precision=s["precision"])
        return out

    def _draft_verification(self, resp: ChatResponse):
        """Claim verification of the DRAFT answer. In-pipeline result when verification ran; otherwise a post-hoc
        measurement with the same verifier (it never changes the answer that was returned)."""
        if resp.verification is not None:
            return resp.verification if not resp.verification.error else None
        if not (self.measure_draft and self.verifier and resp.status == "answered" and resp.citations):
            return None
        n = resp.trace.get("context_chunks", len(resp.evidence))
        gen = GenerationResult(resp.answer, "answered", citations=resp.citations, used_chunks=resp.evidence[:n],
                               marked_answer=resp.trace.get("marked_answer", ""))
        try:
            return self.verifier.verify(gen)[0]
        except Exception:  # noqa: BLE001
            logger.exception("Post-hoc verification failed")
            return None

    def _verification(self, resp: ChatResponse, q: EvalQuestion) -> dict:
        ver = self._draft_verification(resp)
        out = {k: NAN for k in ("claims_total", "claims_supported", "claims_contradicted", "claims_insufficient",
                                "citations_total", "citations_valid", "claims_with_valid_citation")}
        out["verification_source"] = "pipeline" if resp.verification is not None else ("posthoc" if ver else "")
        if ver is not None:
            c = ver.counts
            out.update(claims_total=c["total_claims"], claims_supported=c["supported"], claims_contradicted=c["contradicted"],
                       claims_insufficient=c["insufficient"], citations_total=c["citations_total"],
                       citations_valid=c["citations_valid"], claims_with_valid_citation=c["claims_with_valid_citation"])
            out["claims_json"] = [
                {"claim": v.claim.text, "system_label": v.verdict.value,
                 "evidence": v.supporting.window_text if v.supporting else (v.evidence[0].window_text if v.evidence else ""),
                 "source": (f"{(v.supporting or v.evidence[0]).filename}, p. {(v.supporting or v.evidence[0]).page_label}" if v.evidence else "")}
                for v in ver.claims
            ]
        # gold citation agreement (final citations vs human-annotated expected citations)
        cites = [{"filename": c.filename, "page_start": c.page_start, "page_end": c.page_end} for c in resp.citations]
        if q.expected_citations and cites:
            s = M.retrieval_scores(cites, q.expected_citations)
            out.update(gold_citation_precision=s["precision"], gold_citation_recall=s["recall"])
        return out

    def _answer_quality(self, resp: ChatResponse, q: EvalQuestion) -> dict:
        out = {"key_fact_recall": NAN, "judge_score": NAN}
        if q.key_facts:
            out["key_fact_recall"] = M.key_fact_recall(resp.answer, q.key_facts)
        if self.judge_llm:
            try:
                out["judge_score"] = judge_answer(self.judge_llm, q.question, q.reference_answer, resp.answer)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Judge failed for %s: %s", q.id, exc)
        return out


# ---------------------------------------------------------------------- aggregation
def summarize(df: pd.DataFrame, exp_id: str) -> dict:
    ok = df[df["status"] != "error"]
    g = lambda col: ok[col] if col in ok else pd.Series(dtype=float)  # noqa: E731
    mean = lambda col, d=None: _nan_if_empty((d if d is not None else ok)[col].mean()) if col in ok else NAN  # noqa: E731
    simple, cplx = ok[ok.gold_query_type == "SIMPLE"], ok[ok.gold_query_type == "COMPLEX"]
    routed = ok[ok.pred_query_type != ""]
    claims = g("claims_total").sum()
    retried = ok[ok.get("correction_retried", pd.Series(False, index=ok.index)) == True]  # noqa: E712
    answerable = ok[ok.answerability != "unanswerable"]
    unans = ok[ok.answerability == "unanswerable"]
    return {
        "experiment": exp_id, "name": EXPERIMENT_NAMES.get(exp_id, exp_id), "n_questions": len(df), "n_errors": int((df["status"] == "error").sum()),
        # adaptive
        "query_type_accuracy": M.accuracy(list(routed.pred_query_type), list(routed.gold_query_type)) if len(routed) else NAN,
        "decomposition_recall": mean("decomp_recall"), "decomposition_precision": mean("decomp_precision"),
        "avg_sub_questions_complex": mean("n_sub_questions", cplx),
        "latency_mean_ms": mean("latency_ms"), "latency_p50_ms": _q(g("latency_ms"), .5), "latency_p95_ms": _q(g("latency_ms"), .95),
        "latency_simple_ms": mean("latency_ms", simple), "latency_complex_ms": mean("latency_ms", cplx),
        "retrieval_ms_simple": mean("retrieval_ms", simple), "retrieval_ms_complex": mean("retrieval_ms", cplx),
        "retrieval_ops_per_query": mean("retrieval_calls"), "retrieval_ops_simple": mean("retrieval_calls", simple),
        "retrieval_ops_complex": mean("retrieval_calls", cplx),
        # corrective
        "retr_precision_before": mean("retr_precision_before"), "retr_precision_after": mean("retr_precision_after"),
        "evidence_recall_before": mean("evidence_recall_before"), "evidence_recall_after": mean("evidence_recall_after"),
        "sufficiency_before": mean("sufficient_before"), "sufficiency_after": mean("sufficient_after"),
        "correction_trigger_rate": _nan_if_empty(len(retried) / len(ok)) if len(ok) else NAN,
        "correction_success_rate": mean("correction_success", retried),
        "attempts_per_unit": mean("attempts_per_unit"),
        # verification (draft answer claims; NLI-estimated unless human-labelled, see README)
        "claims_total": claims,
        "unsupported_claim_rate": M.unsupported_claim_rate(g("claims_supported").sum(), claims) if claims else NAN,
        "contradicted_claim_rate": M.safe_div(g("claims_contradicted").sum(), claims),
        "citation_precision": M.citation_precision(g("citations_valid").sum(), g("citations_total").sum()),
        "citation_coverage": M.citation_coverage(g("claims_with_valid_citation").sum(), claims),
        "gold_citation_precision": mean("gold_citation_precision"), "gold_citation_recall": mean("gold_citation_recall"),
        "abstention_accuracy": mean("abstention_correct"),
        "false_answer_rate_unanswerable": _nan_if_empty(1 - unans["abstained"].mean()) if len(unans) else NAN,
        "over_abstention_rate_answerable": mean("abstained", answerable[answerable.answerability == "full"]),
        # overall
        "key_fact_recall": mean("key_fact_recall"), "judge_score": mean("judge_score"),
        "avg_tokens_in": mean("tokens_in"), "avg_tokens_out": mean("tokens_out"), "avg_llm_calls": mean("llm_calls"),
        "fallback_rate": _nan_if_empty((ok["fallbacks"] != "").mean()) if len(ok) else NAN,
    }


def _q(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if len(s.dropna()) else NAN


def _nan_if_empty(x) -> float:
    return NAN if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
