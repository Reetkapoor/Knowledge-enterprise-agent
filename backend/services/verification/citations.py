"""Step 4: citation validation. A citation is valid only if the source exists, its document id and page range
are real, and the cited passage actually supports the claim."""
from __future__ import annotations

from typing import Callable

from backend.schemas.agentic import CitationCheck
from backend.schemas.chat import RetrievedChunk
from backend.schemas.documents import DocumentRecord


class CitationValidator:
    def __init__(self, get_document: Callable[[str], DocumentRecord | None], existing_chunk_ids: Callable[[list[str]], set[str]]):
        self.get_document = get_document
        self.existing_chunk_ids = existing_chunk_ids

    def check_source(self, marker: str, sources: list[RetrievedChunk]) -> tuple[CitationCheck, RetrievedChunk | None]:
        """Existence + page validity only; the caller adds the entailment check."""
        check = CitationCheck(source_id=marker)
        try:
            n = int(marker.upper().lstrip("S"))
        except ValueError:
            check.reason = "malformed source marker"
            return check, None
        if not 1 <= n <= len(sources):
            check.reason = f"source {marker} was not part of the retrieved evidence"
            return check, None
        chunk = sources[n - 1]
        check.chunk_id = chunk.chunk_id
        check.exists = chunk.chunk_id in self.existing_chunk_ids([chunk.chunk_id])
        if not check.exists:
            check.reason = "chunk no longer exists in the index"
            return check, chunk
        doc = self.get_document(chunk.document_id)
        if doc is None:
            check.reason = "document id not found in registry"
        elif not (1 <= chunk.page_start <= chunk.page_end <= doc.num_pages):
            check.reason = f"pages {chunk.page_start}-{chunk.page_end} outside document ({doc.num_pages} pages)"
        else:
            check.page_valid = True
        return check, chunk
