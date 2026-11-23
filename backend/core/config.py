"""Central configuration. Every tunable comes from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM provider
    llm_provider: str = "anthropic"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "claude-sonnet-5-5"
    llm_base_url: str = ""
    llm_timeout_seconds: float = 60.0
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = 32

    # Chunking (token counts are estimated as words * tokens_per_word)
    chunk_size: int = 500
    chunk_overlap: int = 100
    tokens_per_word: float = 1.3

    # Retrieval
    top_k: int = 5
    min_similarity: float = 0.35
    max_context_tokens: int = 3500

    # Adaptive RAG
    max_sub_questions: int = 3

    # Corrective RAG
    max_corrective_retries: int = 1  # spec: ONE retry; 0 disables retrying
    corrective_partial_threshold: float = 0.5  # sufficient but <= this fraction relevant -> PARTIALLY_RELEVANT
    corrective_retry_on: str = "IRRELEVANT,INSUFFICIENT_EVIDENCE"

    # Claim-level verification (NLI)
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_batch_size: int = 16
    nli_max_length: int = 512
    nli_entail_threshold: float = 0.60  # uncalibrated default; tune with scripts/calibrate_nli.py
    nli_contradict_threshold: float = 0.60
    nli_window_sentences: int = 3  # evidence is split into sentence windows (NLI models are length-limited)
    nli_window_stride: int = 2
    nli_top_windows: int = 4  # windows per claim passed to the NLI model
    numeric_guard: bool = True  # demote SUPPORTED if claim numbers are absent from the supporting window
    max_claims: int = 12

    # Storage
    chroma_dir: Path = Path("data/chroma_db")
    upload_dir: Path = Path("data/uploads")
    registry_path: Path = Path("data/documents.json")

    # Limits
    max_upload_mb: int = 25
    max_question_chars: int = 2000
    history_turns: int = 6

    @field_validator("chroma_dir", "upload_dir", "registry_path", mode="after")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else BASE_DIR / v

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.chunk_size < 20:
            raise ValueError("CHUNK_SIZE must be >= 20")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and < CHUNK_SIZE")
        if self.top_k < 1:
            raise ValueError("TOP_K must be >= 1")
        if self.tokens_per_word <= 0:
            raise ValueError("TOKENS_PER_WORD must be > 0")
        return self

    @property
    def corrective_retry_verdicts(self) -> set[str]:
        return {v.strip().upper() for v in self.corrective_retry_on.split(",") if v.strip()}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
