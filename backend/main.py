"""FastAPI entrypoint.  Run:  uvicorn backend.main:app --reload"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api import chat, documents
from backend.core.config import Settings, get_settings
from backend.core.container import Container
from backend.schemas.documents import DocumentStatus, HealthResponse
from backend.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def create_app(container: Container | None = None, settings: Settings | None = None) -> FastAPI:
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if container is None:
            app.state.container = Container.build(settings or get_settings())
        else:
            app.state.container = container
        try:
            app.state.container.embedder.warm_up()
        except Exception as exc:  # non-fatal: uploads/chat will report the problem
            logger.warning("Embedding model warm-up failed: %s", exc)
        try:
            app.state.container.nli.warm_up()
        except Exception as exc:  # non-fatal: baseline still works; verification falls back with a recorded error
            logger.warning("NLI model warm-up failed (verification will be unavailable): %s", exc)
        if not app.state.container.settings.llm_configured:
            logger.warning("LLM_API_KEY is not set; /chat will return 503 until it is configured.")
        yield

    app = FastAPI(
        title="Knowledge Enterprise Agent",
        version="0.2.0",
        description="Upload PDFs, ask questions, get answers with document/page citations. Baseline RAG or full Adaptive + Corrective + Verified RAG.",
        lifespan=lifespan,
    )
    app.include_router(documents.router)
    app.include_router(chat.router)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        c = app.state.container
        docs = c.registry.list()
        db = c.vector_store.health()
        return HealthResponse(
            status="ok" if db == "ok" and c.settings.llm_configured else "degraded",
            vector_db=db,
            indexed_documents=sum(d.status == DocumentStatus.INDEXED for d in docs),
            failed_documents=sum(d.status == DocumentStatus.FAILED for d in docs),
            total_chunks=c.vector_store.count(),
            embedding_model=c.embedder.model_name,
            embedding_model_loaded=c.embedder.is_loaded,
            llm_provider=c.settings.llm_provider,
            llm_model=c.settings.llm_model,
            llm_configured=c.settings.llm_configured,
        )

    return app


app = create_app()
