"""Claim-level answer verification: extract claims -> match evidence -> NLI -> validate citations -> correct.

The LLM only *extracts* claims. The verdicts come from a separate pretrained NLI model, so the generator never
grades its own answer.
"""
from __future__ import annotations

import logging
from collections import Counter

from backend.schemas.agentic import (
    CitationCheck,
    Claim,
    ClaimVerdict,
    EvidenceMatch,
    VerificationResult,
    VerifiedClaim,
)
from backend.schemas.chat import RetrievedChunk
from backend.services.answer_service import GenerationResult
from backend.services.embedding_service import BaseEmbedder
from backend.services.verification.citations import CitationValidator
from backend.services.verification.claims import ClaimExtractor
from backend.services.verification.corrector import CorrectionOutcome, correct_answer
from backend.services.verification.evidence import EvidenceIndex, Window, missing_numbers
from backend.services.verification.nli import NLIModel, NLIScores

logger = logging.getLogger(__name__)


def decide_verdict(
    best_entail: float, best_contra: float, entail_threshold: float, contradict_threshold: float
) -> tuple[ClaimVerdict, bool]:
    """Pure threshold rule (unit-tested and calibrated offline). Returns (verdict, conflict)."""
    ent_ok, con_ok = best_entail >= entail_threshold, best_contra >= contradict_threshold
    if ent_ok and con_ok:
        return (ClaimVerdict.SUPPORTED if best_entail >= best_contra else ClaimVerdict.CONTRADICTED), True
    if ent_ok:
        return ClaimVerdict.SUPPORTED, False
    if con_ok:
        return ClaimVerdict.CONTRADICTED, False
    return ClaimVerdict.INSUFFICIENT, False


def judge_claim(
    claim: str, scored: list[tuple[str, NLIScores]], entail_t: float, contradict_t: float, numeric_guard: bool = True
) -> dict:
    """Aggregate NLI scores of one claim against several evidence windows into a verdict.

    Shared by the live verifier and by scripts/eval_verifier.py + scripts/calibrate_nli.py, so the evaluated
    decision rule is exactly the deployed one. `scored` = [(window_text, scores)]."""
    best_e = max(scored, key=lambda x: x[1].entailment, default=None)
    best_c = max(scored, key=lambda x: x[1].contradiction, default=None)
    e = best_e[1].entailment if best_e else 0.0
    k = best_c[1].contradiction if best_c else 0.0
    verdict, conflict = decide_verdict(e, k, entail_t, contradict_t)
    demoted = False
    if verdict == ClaimVerdict.SUPPORTED and numeric_guard and best_e and missing_numbers(claim, best_e[0]):
        verdict, demoted, conflict = ClaimVerdict.INSUFFICIENT, True, False
    return {"verdict": verdict, "conflict": conflict, "demoted": demoted, "entailment": e, "contradiction": k}


class AnswerVerifier:
    def __init__(
        self, extractor: ClaimExtractor, nli: NLIModel, embedder: BaseEmbedder, validator: CitationValidator, *,
        entail_threshold: float = 0.6, contradict_threshold: float = 0.6, top_windows: int = 4,
        window_sentences: int = 3, window_stride: int = 2, numeric_guard: bool = True,
    ):
        self.extractor, self.nli, self.embedder, self.validator = extractor, nli, embedder, validator
        self.entail_t, self.contra_t = entail_threshold, contradict_threshold
        self.top_windows, self.window_sentences, self.window_stride = top_windows, window_sentences, window_stride
        self.numeric_guard = numeric_guard

    # ------------------------------------------------------------------ public
    def verify(self, generation: GenerationResult) -> tuple[VerificationResult, CorrectionOutcome]:
        sources = generation.used_chunks
        claims, unavailable = self.extractor.extract(generation.marked_answer or generation.answer, len(sources))
        base = dict(
            original_answer=generation.answer, unavailable_statements=unavailable, nli_model=self.nli.model_name,
            thresholds={"entail": self.entail_t, "contradict": self.contra_t},
        )
        if not claims:
            # Nothing factual to check (e.g. the answer only says information is unavailable).
            res = VerificationResult(final_answer=generation.answer, counts=_counts([]), **base)
            return res, CorrectionOutcome(generation.answer, False, [c.chunk_id for c in generation.citations], unchanged=True)

        index = EvidenceIndex(self.embedder, sources, self.window_sentences, self.window_stride)
        plan = [self._plan(claim, index, sources) for claim in claims]

        pairs = list({(w.text, claim.text) for claim, wins, _ in plan for w in wins})  # one batched NLI call
        scores = dict(zip(pairs, self.nli.predict(pairs))) if pairs else {}
        verified = [self._verify_claim(c, wins, cited, index, scores, sources) for c, wins, cited in plan]

        outcome = correct_answer(generation.answer, verified, unavailable, sources, self.contra_t)
        res = VerificationResult(
            final_answer=outcome.final_answer, claims=verified, abstained=outcome.abstained,
            counts=_counts(verified), **base,
        )
        return res, outcome

    # ------------------------------------------------------------------ internals
    def _plan(self, claim: Claim, index: EvidenceIndex, sources: list[RetrievedChunk]):
        """Windows to score for this claim: globally most similar ones + the best windows of each cited chunk."""
        wins: dict[tuple[str, str], Window] = {}
        for w, _ in index.top(claim.text, self.top_windows):
            wins[(w.chunk.chunk_id, w.text)] = w
        cited: dict[str, tuple[CitationCheck, RetrievedChunk | None]] = {}
        for marker in claim.cited_sources:
            check, chunk = self.validator.check_source(marker, sources)
            cited[marker] = (check, chunk)
            if chunk is not None and check.exists:
                for w, _ in index.top(claim.text, 2, chunk.chunk_id):
                    wins[(w.chunk.chunk_id, w.text)] = w
        return claim, list(wins.values()), cited

    def _verify_claim(self, claim, wins, cited, index, scores: dict[tuple[str, str], NLIScores], sources) -> VerifiedClaim:
        matches: list[EvidenceMatch] = []
        sim_cache = {(w.chunk.chunk_id, w.text): s for w, s in index.top(claim.text, len(index.windows))}
        for w in wins:
            sc = scores[(w.text, claim.text)]
            c = w.chunk
            matches.append(EvidenceMatch(
                chunk_id=c.chunk_id, document_id=c.document_id, filename=c.filename, page_start=c.page_start,
                page_end=c.page_end, page_label=c.page_label, window_text=w.text,
                similarity=round(sim_cache.get((c.chunk_id, w.text), 0.0), 4),
                entailment=round(sc.entailment, 4), neutral=round(sc.neutral, 4), contradiction=round(sc.contradiction, 4),
            ))
        matches.sort(key=lambda m: -m.entailment)
        best_e = max(matches, key=lambda m: m.entailment, default=None)
        best_c = max(matches, key=lambda m: m.contradiction, default=None)
        j = judge_claim(
            claim.text, [(m.window_text, NLIScores(m.entailment, m.neutral, m.contradiction)) for m in matches],
            self.entail_t, self.contra_t, self.numeric_guard,
        )
        verdict, conflict, demoted, e, k = j["verdict"], j["conflict"], j["demoted"], j["entailment"], j["contradiction"]

        vc = VerifiedClaim(
            claim=claim, verdict=verdict, entailment=e, contradiction=k, conflict=conflict,
            demoted_by_numeric_guard=demoted, evidence=matches,
            supporting=best_e if verdict == ClaimVerdict.SUPPORTED else None,
            contradicting=best_c if (verdict == ClaimVerdict.CONTRADICTED or conflict) else None,
        )

        # Citation validation: exists + valid pages + the cited passage itself entails the claim.
        for marker, (check, chunk) in cited.items():
            if chunk is not None and check.exists and check.page_valid:
                own = [m for m in matches if m.chunk_id == chunk.chunk_id]
                ok = [m for m in own if m.entailment >= self.entail_t
                      and not (self.numeric_guard and missing_numbers(claim.text, m.window_text))]
                check.supports_claim = bool(ok)
                check.reason = check.reason or ("" if ok else "cited passage does not entail the claim")
            vc.citation_checks.append(check)
        vc.final_citations = list(dict.fromkeys(c.chunk_id for c in vc.citation_checks if c.valid and c.chunk_id))

        if verdict == ClaimVerdict.SUPPORTED and not vc.final_citations and vc.supporting is not None:
            # Replace missing/invalid citations with the best supporting retrieved chunk (metadata, never invented).
            marker = next((f"S{i}" for i, s in enumerate(sources, 1) if s.chunk_id == vc.supporting.chunk_id), None)
            if marker:
                check, _ = self.validator.check_source(marker, sources)
                if check.exists and check.page_valid:
                    vc.final_citations, vc.reattributed = [vc.supporting.chunk_id], True
        return vc


def _counts(claims: list[VerifiedClaim]) -> dict[str, int]:
    c = Counter(v.verdict.value for v in claims)
    checks = [ck for v in claims for ck in v.citation_checks]
    return {
        "total_claims": len(claims),
        "supported": c[ClaimVerdict.SUPPORTED.value],
        "contradicted": c[ClaimVerdict.CONTRADICTED.value],
        "insufficient": c[ClaimVerdict.INSUFFICIENT.value],
        "citations_total": len(checks),
        "citations_valid": sum(ck.valid for ck in checks),
        "claims_with_valid_citation": sum(any(ck.valid for ck in v.citation_checks) for v in claims),  # original citations
        "claims_with_final_citation": sum(bool(v.final_citations) for v in claims),  # after re-attribution
        "reattributed": sum(v.reattributed for v in claims),
    }
