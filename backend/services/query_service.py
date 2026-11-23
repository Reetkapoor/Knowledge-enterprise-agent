"""Query processing: turns the user's message into a standalone retrieval query.

Adaptive RAG (complexity routing, decomposition) will extend this class later.
"""
from __future__ import annotations

import logging

from backend.schemas.chat import ProcessedQuery, Turn
from backend.services.llm_service import LLMError, LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You rewrite a user's latest chat message into a single standalone search question. "
    "Use the conversation only to resolve pronouns and references (e.g. 'they', 'it', 'that policy'). "
    "Do NOT answer the question and do NOT add facts. If the message is already standalone, return it unchanged. "
    "Output only the question text."
)


class QueryProcessor:
    def __init__(self, llm: LLMProvider, history_turns: int = 6):
        self.llm = llm
        self.history_turns = history_turns

    def process(self, question: str, history: list[Turn]) -> ProcessedQuery:
        if not history:
            return ProcessedQuery(original=question, standalone=question)
        convo = "\n".join(
            f"User: {t.question}\nAssistant: {t.answer[:500]}" for t in history[-self.history_turns :]
        )
        prompt = f"CONVERSATION:\n{convo}\n\nLATEST MESSAGE: {question}\n\nStandalone question:"
        try:
            out = self.llm.generate(_SYSTEM, prompt, max_tokens=200)
        except LLMError as exc:
            if exc.not_configured:
                raise
            logger.warning("Query rewrite failed, using original question: %s", exc)
            return ProcessedQuery(original=question, standalone=question, metadata={"rewrite_error": str(exc)})
        standalone = out.strip().splitlines()[0].strip().strip('"') if out.strip() else ""
        if not standalone or len(standalone) > 500:
            standalone = question
        return ProcessedQuery(
            original=question, standalone=standalone, rewritten=standalone != question
        )
