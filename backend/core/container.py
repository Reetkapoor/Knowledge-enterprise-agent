"""Dependency wiring. Tests inject fake embedder / LLM through build()."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.core.config import Settings
from backend.services.answer_service import AnswerGenerator
from backend.services.conversation_store import ConversationStore
from backend.services.document_registry import DocumentRegistry
from backend.services.document_service import DocumentService
from backend.services.embedding_service import BaseEmbedder, SentenceTransformerEmbedder
from backend.services.llm_service import LLMProvider, build_llm_provider
from backend.services.corrective_rag import CorrectiveRetriever, QueryRefiner, RetrievalEvaluator
from backend.services.query_router import QueryRouter
from backend.services.query_service import QueryProcessor
from backend.services.rag_pipeline import RAGPipeline
from backend.services.retrieval_service import RetrievalService
from backend.services.vector_store import VectorStore
from backend.services.verification.citations import CitationValidator
from backend.services.verification.claims import ClaimExtractor
from backend.services.verification.nli import HFNLIModel, NLIModel
from backend.services.verification.verifier import AnswerVerifier

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    embedder: BaseEmbedder
    vector_store: VectorStore
    registry: DocumentRegistry
    documents: DocumentService
    pipeline: RAGPipeline
    conversations: ConversationStore
    nli: NLIModel
    llm: LLMProvider

    @classmethod
    def build(
        cls, settings: Settings, embedder: BaseEmbedder | None = None, llm: LLMProvider | None = None,
        nli: NLIModel | None = None,
    ) -> "Container":
        embedder = embedder or SentenceTransformerEmbedder(
            settings.embedding_model, settings.embedding_batch_size, settings.embedding_query_prefix
        )
        llm = llm or build_llm_provider(settings)
        vector_store = VectorStore(settings.chroma_dir)
        registry = DocumentRegistry(settings.registry_path)
        conversations = ConversationStore(max_turns=settings.history_turns)
        nli = nli or HFNLIModel(settings.nli_model, settings.nli_batch_size, settings.nli_max_length)
        retriever = RetrievalService(embedder, vector_store, settings.top_k, settings.min_similarity)
        corrective = CorrectiveRetriever(
            retriever, RetrievalEvaluator(llm, settings.corrective_partial_threshold), QueryRefiner(llm),
            settings.max_corrective_retries, settings.corrective_retry_verdicts,
        )
        verifier = AnswerVerifier(
            ClaimExtractor(llm, settings.max_claims), nli, embedder,
            CitationValidator(registry.get, vector_store.existing_ids),
            entail_threshold=settings.nli_entail_threshold, contradict_threshold=settings.nli_contradict_threshold,
            top_windows=settings.nli_top_windows, window_sentences=settings.nli_window_sentences,
            window_stride=settings.nli_window_stride, numeric_guard=settings.numeric_guard,
        )
        pipeline = RAGPipeline(
            QueryProcessor(llm, settings.history_turns),
            retriever,
            AnswerGenerator(llm, settings.tokens_per_word, settings.max_context_tokens),
            conversations,
            vector_store,
            router=QueryRouter(llm, settings.max_sub_questions),
            corrective=corrective,
            verifier=verifier,
            usage_source=llm,
        )
        stale = {d.embedding_model for d in registry.list() if d.embedding_model not in (None, embedder.model_name)}
        if stale:
            logger.warning(
                "Documents were indexed with different embedding model(s) %s than the current '%s'. "
                "Re-index them or retrieval quality will be wrong.", sorted(stale), embedder.model_name,
            )
        return cls(settings, embedder, vector_store, registry,
                   DocumentService(settings, registry, vector_store, embedder), pipeline, conversations, nli, llm)
