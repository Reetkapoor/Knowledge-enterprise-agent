"""Deterministic fakes for the agentic components (no network, no model downloads)."""
from __future__ import annotations

import json
import re

from backend.services.llm_service import LLMProvider
from backend.services.verification.nli import NLIModel, NLIScores

_STOP = {"the", "a", "an", "of", "to", "is", "are", "and", "in", "for", "per", "be", "can", "may", "must", "that", "this", "it"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


class ScriptedLLM(LLMProvider):
    """Dispatches on the system prompt. Override any stage with a callable(user) -> str (or a dict -> JSON)."""

    KEYS = {
        "router": "route questions",
        "evaluator": "judge document retrieval",
        "refiner": "rewrite a failed document search query",
        "claims": "atomic factual claims",
        "answer": "enterprise document assistant",
        "rewrite": "standalone search question",
    }

    def __init__(self, **stages):
        self.stages = stages
        self.calls: list[tuple[str, str, str]] = []  # (stage, system, user)
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def generate(self, system, user, *, max_tokens=None):
        stage = next((k for k, needle in self.KEYS.items() if needle in system), "unknown")
        self.calls.append((stage, system, user))
        self.usage["calls"] += 1
        self.usage["input_tokens"] += len(user.split())
        handler = self.stages.get(stage)
        out = handler(user) if callable(handler) else handler
        if out is None:
            raise AssertionError(f"no scripted response for stage {stage!r}")
        return out if isinstance(out, str) else json.dumps(out)

    def stage_calls(self, stage: str) -> list[str]:
        return [u for s, _, u in self.calls if s == stage]


class FakeNLI(NLIModel):
    """Entailment if (almost) all claim tokens occur in the premise; contradiction via explicit rules."""

    model_name = "fake-nli"

    def __init__(self, contradictions: list[tuple[str, str]] | None = None):
        # (premise substring, hypothesis substring) pairs that should be scored as contradiction
        self.contradictions = contradictions or []
        self.pairs_seen: list[tuple[str, str]] = []

    def predict(self, pairs):
        out = []
        for premise, hyp in pairs:
            self.pairs_seen.append((premise, hyp))
            if any(p in premise and h in hyp for p, h in self.contradictions):
                out.append(NLIScores(0.03, 0.07, 0.90))
                continue
            ht, pt = _tokens(hyp), _tokens(premise)
            overlap = len(ht & pt) / max(1, len(ht))
            out.append(NLIScores(0.92, 0.06, 0.02) if overlap >= 0.8 else NLIScores(0.10, 0.85, 0.05))
        return out
