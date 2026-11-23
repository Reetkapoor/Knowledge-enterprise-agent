"""Step 2: evidence matching. Retrieved chunks are too long for NLI models (512 tokens), so each chunk is
split into short sentence windows; for every claim the most similar windows become NLI premises.
Document id, filename and page range stay attached to every window."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from backend.schemas.chat import RetrievedChunk
from backend.services.embedding_service import BaseEmbedder

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}")


@dataclass(frozen=True)
class Window:
    chunk: RetrievedChunk
    text: str


def split_sentences(text: str, max_words: int = 60) -> list[str]:
    out: list[str] = []
    for part in _SENT.split(text):
        words = part.split()
        if not words:
            continue
        for i in range(0, len(words), max_words):  # run-on text without punctuation
            out.append(" ".join(words[i : i + max_words]))
    return out


def window_texts(text: str, sentences: int = 3, stride: int = 2) -> list[str]:
    sents = split_sentences(text)
    out: list[str] = []
    for i in range(0, max(1, len(sents) - sentences + stride), stride):
        w = " ".join(sents[i : i + sentences])
        if w and w not in out:
            out.append(w)
        if i + sentences >= len(sents):
            break
    return out


def make_windows(chunk: RetrievedChunk, sentences: int = 3, stride: int = 2) -> list[Window]:
    return [Window(chunk, t) for t in window_texts(chunk.text, sentences, stride)]


class EvidenceIndex:
    """Windows + embeddings for one verification call (built once, queried per claim)."""

    def __init__(self, embedder: BaseEmbedder, chunks: list[RetrievedChunk], sentences: int = 3, stride: int = 2):
        self.embedder = embedder
        self.windows: list[Window] = []
        for c in chunks:
            self.windows.extend(make_windows(c, sentences, stride))
        # a chunk that is one short blob still needs one window
        self._vecs = embedder.embed_documents([w.text for w in self.windows]) if self.windows else np.zeros((0, 0))

    def top(self, claim: str, n: int, chunk_id: str | None = None) -> list[tuple[Window, float]]:
        """Most similar windows to the claim, optionally restricted to one chunk."""
        if not self.windows:
            return []
        q = np.asarray(self.embedder.embed_query(claim), dtype=np.float32)
        sims = self._vecs @ q
        order = np.argsort(-sims)
        out: list[tuple[Window, float]] = []
        for i in order:
            w = self.windows[int(i)]
            if chunk_id is None or w.chunk.chunk_id == chunk_id:
                out.append((w, float(sims[int(i)])))
                if len(out) >= n:
                    break
        return out


# ---- numeric guard -----------------------------------------------------------------------------
_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "ninety": 90, "hundred": 100,
}
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?|\b(?:" + "|".join(_WORD_NUM) + r")\b", re.IGNORECASE)


def numbers_in(text: str) -> set[str]:
    out: set[str] = set()
    for m in _NUM.findall(text):
        m = m.lower()
        if m in _WORD_NUM:
            out.add(str(_WORD_NUM[m]))
        else:
            m = m.replace(",", "")
            out.add(m[:-2] if m.endswith(".0") else m)
    return out


def missing_numbers(claim: str, evidence: str) -> set[str]:
    return numbers_in(claim) - numbers_in(evidence)
