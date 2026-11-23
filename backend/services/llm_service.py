"""LLM provider layer. RAG code only depends on LLMProvider.generate()."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import httpx

from backend.core.config import Settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Provider failure with a message that is safe to show (never contains credentials)."""

    def __init__(self, message: str, *, not_configured: bool = False):
        super().__init__(message)
        self.not_configured = not_configured


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, *, max_tokens: int | None = None) -> str: ...

    def usage_snapshot(self) -> dict[str, int]:
        """Cumulative {calls, input_tokens, output_tokens}; providers that don't track usage return zeros."""
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, **getattr(self, "usage", {})}


class _HTTPProvider(LLMProvider):
    default_base_url = ""

    def __init__(self, settings: Settings):
        self.model = settings.llm_model
        self.base_url = (settings.llm_base_url or self.default_base_url).rstrip("/")
        self.timeout = settings.llm_timeout_seconds
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature
        self._key = settings.llm_api_key.get_secret_value().strip()
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def generate(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        if not self._key:
            raise LLMError("LLM_API_KEY is not configured on the server.", not_configured=True)
        url, headers, payload = self._request(system, user, max_tokens or self.max_tokens)
        last = "unknown error"
        for attempt in range(3):
            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
            except httpx.TimeoutException:
                last = "request timed out"
            except httpx.HTTPError as exc:
                last = f"network error ({type(exc).__name__})"
            else:
                if resp.status_code == 200:
                    try:
                        body = resp.json()
                        self._record_usage(body)
                        return self._parse(body)
                    except Exception as exc:
                        raise LLMError("LLM returned an unexpected response format.") from exc
                if resp.status_code in (401, 403):
                    raise LLMError("LLM rejected the API key (authentication failed).")
                last = f"HTTP {resp.status_code}"
                if resp.status_code not in (408, 409, 429) and resp.status_code < 500:
                    raise LLMError(f"LLM request failed ({last}): {self._error_detail(resp)}")
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM service unavailable after retries ({last}).")

    def _record_usage(self, body: dict) -> None:
        u = body.get("usage") or {}
        self.usage["calls"] += 1
        self.usage["input_tokens"] += int(u.get("input_tokens", u.get("prompt_tokens", 0)) or 0)
        self.usage["output_tokens"] += int(u.get("output_tokens", u.get("completion_tokens", 0)) or 0)

    @staticmethod
    def _error_detail(resp: httpx.Response) -> str:
        try:
            err = resp.json().get("error", {})
            return str(err.get("message", ""))[:200] if isinstance(err, dict) else str(err)[:200]
        except Exception:
            return ""

    def _request(self, system: str, user: str, max_tokens: int): ...
    def _parse(self, data: dict) -> str: ...


class AnthropicProvider(_HTTPProvider):
    default_base_url = "https://api.anthropic.com"

    def _request(self, system, user, max_tokens):
        return (
            f"{self.base_url}/v1/messages",
            {"x-api-key": self._key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )

    def _parse(self, data):
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()


class OpenAICompatibleProvider(_HTTPProvider):
    default_base_url = "https://api.openai.com/v1"

    def _request(self, system, user, max_tokens):
        return (
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self._key}", "content-type": "application/json"},
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )

    def _parse(self, data):
        return (data["choices"][0]["message"]["content"] or "").strip()


def build_llm_provider(settings: Settings) -> LLMProvider:
    providers = {"anthropic": AnthropicProvider, "openai": OpenAICompatibleProvider}
    try:
        return providers[settings.llm_provider.lower()](settings)
    except KeyError:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Supported: {', '.join(providers)}"
        ) from None
