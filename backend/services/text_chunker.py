"""Text cleaning and page-aware chunking.

Chunk size/overlap are in *approximate tokens* (words * tokens_per_word) so no tokenizer
dependency is needed. Every word carries its source page, so a chunk spanning pages
records the exact page range instead of a single (wrong) page.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from backend.schemas.documents import Chunk, PageText

_SENTENCE_END = (".", "?", "!", ":", ";")


def clean_text(raw: str) -> str:
    """Normalize unicode and whitespace; keep blank-line paragraph boundaries."""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\u00ad", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join words hyphenated across lines
    text = re.sub(r"[ \t\f\v\u200b]+", " ", text)
    paragraphs = (" ".join(p.split()) for p in re.split(r"\n\s*\n", text))
    return "\n\n".join(p for p in paragraphs if p)


@dataclass(slots=True)
class _Word:
    text: str
    page: int
    para_end: bool


def chunk_pages(
    pages: list[PageText],
    *,
    document_id: str,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    tokens_per_word: float = 1.3,
) -> list[Chunk]:
    max_words = max(1, int(chunk_size / tokens_per_word))
    overlap_words = min(int(chunk_overlap / tokens_per_word), max_words - 1)

    words: list[_Word] = []
    for page in pages:
        cleaned = clean_text(page.text)
        for para in cleaned.split("\n\n") if cleaned else []:
            tokens = para.split()
            for i, tok in enumerate(tokens):
                words.append(_Word(tok, page.page_number, i == len(tokens) - 1))

    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        if end < len(words):
            end = _snap_to_boundary(words, start, end)
        window = words[start:end]
        page_list = sorted({w.page for w in window})
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}:{index:05d}",
                document_id=document_id,
                filename=filename,
                chunk_index=index,
                page_start=page_list[0],
                page_end=page_list[-1],
                pages=page_list,
                text=_render(window),
            )
        )
        if end >= len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def _snap_to_boundary(words: list[_Word], start: int, end: int) -> int:
    """Prefer ending at a paragraph/sentence boundary within the last 30% of the window."""
    floor = start + int((end - start) * 0.7)
    for i in range(end - 1, floor - 1, -1):
        if words[i].para_end or words[i].text.endswith(_SENTENCE_END):
            return i + 1
    return end


def _render(window: list[_Word]) -> str:
    out: list[str] = []
    for i, w in enumerate(window):
        out.append(w.text)
        out.append("\n\n" if w.para_end and i < len(window) - 1 else " ")
    return "".join(out).strip()
