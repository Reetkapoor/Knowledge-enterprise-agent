from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from backend.schemas.agentic import MechanismFlags
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.embedding_service import EmbeddingError
from backend.services.llm_service import LLMError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    c = request.app.state.container
    if len(body.question) > c.settings.max_question_chars:
        raise HTTPException(422, f"Question exceeds {c.settings.max_question_chars} characters.")
    try:
        flags = MechanismFlags.for_mode(body.mode, body.flags)
        return c.pipeline.ask(body.question, body.session_id, body.top_k, body.document_ids, flags, body.mode.value)
    except LLMError as exc:
        logger.warning("LLM failure: %s", exc)
        raise HTTPException(503 if exc.not_configured else 502, f"LLM error: {exc}") from exc
    except EmbeddingError as exc:
        logger.warning("Embedding failure: %s", exc)
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled chat error")
        raise HTTPException(500, "Internal error while answering the question.") from exc


@router.delete("/chat/{session_id}")
def clear_conversation(session_id: str, request: Request) -> dict:
    return {"cleared": request.app.state.container.conversations.clear(session_id)}
