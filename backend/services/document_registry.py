"""Document metadata registry persisted as JSON (separate from the vector index)."""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from backend.schemas.documents import DocumentRecord, DocumentStatus

logger = logging.getLogger(__name__)


class DocumentRegistry:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._docs: dict[str, DocumentRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._docs = {d["document_id"]: DocumentRecord(**d) for d in raw}
        except Exception:
            logger.exception("Registry file unreadable; starting empty (backing up the old file)")
            self._path.rename(self._path.with_suffix(".corrupt"))
            return
        # A crash mid-ingestion leaves 'processing' records behind; surface them honestly.
        changed = False
        for rec in self._docs.values():
            if rec.status == DocumentStatus.PROCESSING:
                rec.status, rec.error, changed = (
                    DocumentStatus.FAILED,
                    "Processing was interrupted (server restarted). Please re-upload.",
                    True,
                )
        if changed:
            self._save()

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([d.model_dump(mode="json") for d in self._docs.values()], indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._docs.get(document_id)

    def upsert(self, record: DocumentRecord) -> None:
        with self._lock:
            self._docs[record.document_id] = record
            self._save()

    def delete(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self._docs:
                return False
            del self._docs[document_id]
            self._save()
            return True

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.upload_timestamp, reverse=True)
