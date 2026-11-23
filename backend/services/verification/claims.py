"""Step 1: extract atomic factual claims (with the [S#] markers the generator attached) from an answer."""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from backend.schemas.agentic import Claim
from backend.services.llm_json import generate_structured
from backend.services.llm_service import LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM = """You split an answer into atomic factual claims for fact checking.

Rules:
1. Each claim is ONE self-contained factual statement (resolve pronouns so it can be checked alone). Split sentences
   that contain several facts. Keep numbers, durations, names and conditions exactly as written. Do not add facts.
2. For each claim copy the source markers (like S1, S3) that appear on the sentence it came from into "sources"
   (just the labels, e.g. ["S1"]). If the sentence has no marker use [].
3. Statements saying that information is NOT available / NOT covered in the documents are not claims; copy them
   word for word into "unavailable_statements" instead.
4. Skip greetings, hedges and meta text. Return at most {max_claims} claims.

Reply with one JSON object and nothing else:
{{"claims": [{{"text": "...", "sources": ["S1"]}}], "unavailable_statements": []}}"""


class _ClaimOut(BaseModel):
    text: str
    sources: list[str] = Field(default_factory=list)


class _ExtractOut(BaseModel):
    claims: list[_ClaimOut] = Field(default_factory=list)
    unavailable_statements: list[str] = Field(default_factory=list)


class ClaimExtractor:
    def __init__(self, llm: LLMProvider, max_claims: int = 12):
        self.llm = llm
        self.max_claims = max_claims

    def extract(self, marked_answer: str, n_sources: int) -> tuple[list[Claim], list[str]]:
        out = generate_structured(
            self.llm, _SYSTEM.format(max_claims=self.max_claims), f"ANSWER:\n{marked_answer}", _ExtractOut, max_tokens=900
        )
        claims: list[Claim] = []
        seen: set[str] = set()
        for c in out.claims:
            text = " ".join(c.text.split())
            key = text.lower()
            if len(text) < 8 or key in seen:
                continue
            seen.add(key)
            markers = [m.strip().upper() for m in c.sources]  # validity is checked later by the citation validator
            claims.append(Claim(claim_id=f"C{len(claims) + 1}", text=text, cited_sources=list(dict.fromkeys(markers))))
            if len(claims) >= self.max_claims:
                break
        plain = _strip_markers(marked_answer)
        unavailable = [u.strip() for u in out.unavailable_statements if u.strip() and _strip_markers(u).strip() in plain]
        return claims, unavailable


def _strip_markers(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\[S\d+\]", "", text)).strip()
