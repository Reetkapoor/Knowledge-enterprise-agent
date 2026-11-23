"""Embedding generation. Independent of document processing and retrieval."""
from __future__ import annotations

import hashlib
import logging
import threading
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    pass


class BaseEmbedder(ABC):
    model_name: str

    @abstractmethod
    def _encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 vectors, shape (len(texts), dim)."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Batch-embed chunk texts, encoding identical texts only once."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        first: dict[str, str] = {}
        for key, text in zip(keys, texts):
            first.setdefault(key, text)
        position = {key: i for i, key in enumerate(first)}
        vectors = self._encode(list(first.values()))
        return vectors[[position[k] for k in keys]]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0].tolist()

    def warm_up(self) -> None:  # optional hook
        return None

    @property
    def is_loaded(self) -> bool:
        return True


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local Sentence Transformers model, loaded once and shared by ingestion and queries."""

    def __init__(self, model_name: str, batch_size: int = 32, query_prefix: str = ""):
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self._model = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def warm_up(self) -> None:
        self._load()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        logger.info("Loading embedding model %s", self.model_name)
                        self._model = SentenceTransformer(self.model_name)
                    except Exception as exc:
                        raise EmbeddingError(
                            f"Could not load embedding model '{self.model_name}': "
                            f"{str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__}"
                        ) from exc
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        try:
            vecs = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {type(exc).__name__}") from exc
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0].tolist()
