"""Step 5: build the final answer from verified claims only. Deterministic: no LLM rewrite, so the final text can
never contain a statement that was not checked.

  SUPPORTED      -> kept (with validated citations)
  CONTRADICTED   -> replaced by a quote of the contradicting evidence if it is reliable
                    (contradiction >= threshold, no conflicting support), else dropped
  INSUFFICIENT   -> removed, and the answer says how many statements were omitted
  no claim kept  -> abstain
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backend.schemas.agentic import ClaimVerdict, VerifiedClaim
from backend.schemas.chat import RetrievedChunk
from backend.services.answer_service import format_citation

ABSTAIN_MESSAGE = (
    "I could not verify an answer to this question against the uploaded documents, so I am not providing one."
)
_QUOTE_MAX = 400


@dataclass
class CorrectionOutcome:
    final_answer: str
    abstained: bool
    used_chunk_ids: list[str] = field(default_factory=list)
    unchanged: bool = False


def _cite(chunk_ids: list[str], by_id: dict[str, RetrievedChunk]) -> str:
    labels = [format_citation(by_id[c]) for c in dict.fromkeys(chunk_ids) if c in by_id]
    return f" [{'; '.join(labels)}]" if labels else ""


def correct_answer(
    original_answer: str,
    claims: list[VerifiedClaim],
    unavailable: list[str],
    sources: list[RetrievedChunk],
    contradict_threshold: float,
) -> CorrectionOutcome:
    by_id = {c.chunk_id: c for c in sources}
    # Fast path: nothing to change, so keep the model's original fluent wording and citations.
    if claims and all(
        vc.verdict == ClaimVerdict.SUPPORTED and not vc.conflict and not vc.reattributed
        and vc.final_citations and len(vc.final_citations) == len(set(vc.claim.cited_sources))
        for vc in claims
    ):
        for vc in claims:
            vc.action = "kept"
        used = [cid for vc in claims for cid in vc.final_citations]
        return CorrectionOutcome(original_answer, False, list(dict.fromkeys(used)), unchanged=True)

    lines: list[str] = []
    used: list[str] = []
    omitted = 0
    for vc in claims:
        if vc.verdict == ClaimVerdict.SUPPORTED:
            note = ""
            if vc.conflict and vc.contradicting:
                other = vc.contradicting
                note = f" (note: another passage appears to disagree: {other.filename}, p. {other.page_label})"
            lines.append(f"{vc.claim.text}{_cite(vc.final_citations, by_id)}{note}")
            used.extend(vc.final_citations)
            vc.action = "qualified" if note else "kept"
        elif (
            vc.verdict == ClaimVerdict.CONTRADICTED and vc.contradicting is not None and not vc.conflict
            and vc.contradiction >= contradict_threshold and len(vc.contradicting.window_text) <= _QUOTE_MAX
        ):
            ev = vc.contradicting
            lines.append(f"Correction: the documents state “{ev.window_text}” [{ev.filename}, p. {ev.page_label}]")
            used.append(ev.chunk_id)
            vc.action = "corrected"
        else:
            omitted += 1
            vc.action = "removed"

    if not lines:
        return CorrectionOutcome(ABSTAIN_MESSAGE, True)

    body = "\n".join(f"- {ln}" for ln in lines) if len(lines) > 1 else lines[0]
    tail: list[str] = []
    if unavailable:
        tail.append("Not covered by the documents: " + " ".join(unavailable))
    if omitted:
        tail.append(f"{omitted} statement(s) in the draft answer could not be verified against the documents and were omitted.")
    return CorrectionOutcome("\n\n".join([body, *tail]), False, list(dict.fromkeys(used)))
