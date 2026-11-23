"""Adaptive RAG: classify a question and (for COMPLEX ones) decompose it into sub-questions.

Routing is deliberately separate from retrieval: this module only decides *what to retrieve*,
it never touches the vector store.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from backend.schemas.agentic import QueryType, RoutePlan
from backend.services.llm_json import StructuredOutputError, generate_structured
from backend.services.llm_service import LLMError, LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM = """You route questions for an enterprise document assistant that answers ONLY from uploaded company documents.

Classify the user's question as exactly one of:
- SIMPLE: answerable from one fact or one passage (a single lookup, definition, number, rule).
- COMPLEX: needs several separate pieces of evidence: comparisons between entities, multi-part questions,
  questions that combine several conditions/policies, or ones needing facts from different sections.
- CONVERSATIONAL: greetings, thanks, or questions about the assistant itself that are not about document content.

For COMPLEX questions also give 2 to {max_sub} sub_questions. Each sub-question must be independently
searchable (self-contained, no pronouns referring to other sub-questions) and together they must cover the
original question. For SIMPLE and CONVERSATIONAL questions sub_questions must be an empty list.
For CONVERSATIONAL questions put a short, friendly reply in conversational_reply; never state facts about the
company or the documents in it.

Reply with one JSON object and nothing else:
{{"query_type": "SIMPLE|COMPLEX|CONVERSATIONAL", "sub_questions": [], "reasoning": "<one short sentence>", "conversational_reply": null}}"""


class _RouteOut(BaseModel):
    query_type: QueryType
    sub_questions: list[str] = Field(default_factory=list)
    reasoning: str = ""
    conversational_reply: str | None = None


class QueryRouter:
    def __init__(self, llm: LLMProvider, max_sub_questions: int = 3):
        self.llm = llm
        self.max_sub_questions = max(2, max_sub_questions)

    def route(self, question: str) -> RoutePlan:
        try:
            out = generate_structured(
                self.llm, _SYSTEM.format(max_sub=self.max_sub_questions), f"QUESTION: {question}", _RouteOut, max_tokens=400
            )
        except LLMError as exc:
            if exc.not_configured:
                raise
            return self._default(question, f"routing LLM error: {exc}")
        except StructuredOutputError as exc:
            return self._default(question, f"routing output invalid: {exc}")
        return self._normalise(question, out)

    def _normalise(self, question: str, out: _RouteOut) -> RoutePlan:
        subs = _clean_subs(out.sub_questions, question, self.max_sub_questions)
        if out.query_type == QueryType.COMPLEX:
            if len(subs) < 2:  # a "complex" question without usable decomposition is just a single lookup
                return RoutePlan(
                    query_type=QueryType.SIMPLE, reasoning=f"{out.reasoning} (downgraded: <2 usable sub-questions)".strip()
                )
            return RoutePlan(query_type=QueryType.COMPLEX, sub_questions=subs, reasoning=out.reasoning)
        if out.query_type == QueryType.CONVERSATIONAL:
            reply = (out.conversational_reply or "").strip() or (
                "Hi! I answer questions about your uploaded documents. Ask me something about their content."
            )
            return RoutePlan(query_type=QueryType.CONVERSATIONAL, reasoning=out.reasoning, conversational_reply=reply)
        return RoutePlan(query_type=QueryType.SIMPLE, reasoning=out.reasoning)

    @staticmethod
    def _default(question: str, error: str) -> RoutePlan:
        logger.warning("Router fallback to SIMPLE: %s", error)
        return RoutePlan(query_type=QueryType.SIMPLE, reasoning="router failed; default single-step retrieval", error=error)


def _clean_subs(subs: list[str], question: str, limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in subs:
        s = " ".join(str(s).split()).strip()
        key = s.lower().rstrip("?. ")
        if len(s) < 5 or key in seen or key == question.lower().rstrip("?. "):
            continue
        seen.add(key)
        out.append(s)
    return out[:limit]
