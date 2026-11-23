"""Replaceable NLI interface. Premise = evidence window, hypothesis = claim."""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class NLIError(Exception):
    pass


@dataclass(frozen=True)
class NLIScores:
    entailment: float
    neutral: float
    contradiction: float


class NLIModel(ABC):
    model_name: str

    @abstractmethod
    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        """pairs: (premise, hypothesis). Returns softmax probabilities per pair."""

    def warm_up(self) -> None:
        return None


def label_indices(id2label: dict[int, str]) -> dict[str, int]:
    """Map a model's id2label onto entailment/neutral/contradiction (case-insensitive prefix match)."""
    out: dict[str, int] = {}
    for idx, name in id2label.items():
        n = str(name).lower()
        for key, prefix in (("entailment", "entail"), ("neutral", "neutral"), ("contradiction", "contradict")):
            if n.startswith(prefix):
                out[key] = int(idx)
    if len(out) != 3:
        raise NLIError(f"Cannot map NLI labels {dict(id2label)} to entailment/neutral/contradiction")
    return out


def scores_from_logits(logits: np.ndarray, id2label: dict[int, str]) -> list[NLIScores]:
    idx = label_indices(id2label)
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return [NLIScores(float(r[idx["entailment"]]), float(r[idx["neutral"]]), float(r[idx["contradiction"]])) for r in p]


class HFNLIModel(NLIModel):
    """Local HuggingFace sequence-classification NLI model (default: cross-encoder/nli-deberta-v3-small)."""

    def __init__(self, model_name: str, batch_size: int = 16, max_length: int = 512):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._tok = self._model = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        self._load()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        import torch
                        from transformers import AutoModelForSequenceClassification, AutoTokenizer

                        logger.info("Loading NLI model %s", self.model_name)
                        tok = AutoTokenizer.from_pretrained(self.model_name)
                        model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
                        model.to("cuda" if torch.cuda.is_available() else "cpu")
                        label_indices(model.config.id2label)  # fail early on unknown label sets
                        self._tok, self._model = tok, model
                    except NLIError:
                        raise
                    except Exception as exc:
                        raise NLIError(
                            f"Could not load NLI model '{self.model_name}': {str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__}"
                        ) from exc
        return self._tok, self._model

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        if not pairs:
            return []
        import torch

        tok, model = self._load()
        out: list[NLIScores] = []
        for i in range(0, len(pairs), self.batch_size):
            part = pairs[i : i + self.batch_size]
            enc = tok(
                [p for p, _ in part], [h for _, h in part],
                truncation="only_first", max_length=self.max_length, padding=True, return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                logits = model(**enc).logits.float().cpu().numpy()
            out.extend(scores_from_logits(logits, model.config.id2label))
        return out
