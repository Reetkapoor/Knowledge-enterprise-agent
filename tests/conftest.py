from __future__ import annotations

import hashlib
import re

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.core.container import Container
from backend.main import create_app
from backend.services.embedding_service import BaseEmbedder
from backend.services.llm_service import LLMProvider

_STOP = {"the", "a", "an", "of", "to", "is", "are", "and", "in", "for", "per", "be", "can", "what", "how", "do", "does", "that", "this", "must", "may"}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 3:
            return w[: -len(suf)]
    return w


class FakeEmbedder(BaseEmbedder):
    """Deterministic hashed bag-of-words embedder: offline, fast, retrieval is lexical."""

    model_name = "fake-hash-embedder"
    DIM = 512

    def __init__(self):
        self.encoded_texts: list[str] = []

    def _encode(self, texts):
        self.encoded_texts.extend(texts)
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w in _STOP:
                    continue
                idx = int(hashlib.md5(_stem(w).encode()).hexdigest(), 16) % self.DIM
                out[i, idx] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class FakeLLM(LLMProvider):
    """Records calls. Rewrites follow-ups to `rewrite`; answers by citing the top source."""

    def __init__(self, rewrite: str | None = None, handler=None):
        self.calls: list[tuple[str, str]] = []
        self.rewrite = rewrite
        self.handler = handler

    def generate(self, system, user, *, max_tokens=None):
        self.calls.append((system, user))
        if self.handler:
            return self.handler(system, user)
        if "standalone search question" in system:
            return self.rewrite or user.split("LATEST MESSAGE:")[1].split("\n")[0].strip()
        first = re.search(r"\[S1\] \(.*?\)\n(.*?)(\n\n|\n\nQUESTION)", user, re.S)
        snippet = first.group(1).split(".")[0] if first else "found"
        import json
        return json.dumps({"answer": f"{snippet.strip()}. [S1]", "sufficient": True})

    @property
    def answer_calls(self):
        return [c for c in self.calls if "standalone search question" not in c[0]]


def make_pdf(pages: list[str | None]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            rc = page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
            assert rc >= 0, "test PDF text overflowed"
    data = doc.tobytes()
    doc.close()
    return data


LEAVE_PAGES = [
    "Leave Policy overview. This document describes leave rules for all permanent employees of the company.",
    "Annual leave entitlement. Every permanent employee receives 20 days of annual leave per calendar year, "
    "credited on the first of January.",
    "Carry forward rules. Up to 5 unused annual leave days may be carried forward to the next year. "
    "Carried forward days expire on 31 March.",
]
IT_PAGES = [
    "Password policy. Passwords must be at least 14 characters and rotated every 90 days. "
    "Multi-factor authentication is mandatory for VPN access.",
    "Device security. Laptops must use full disk encryption. Lost devices must be reported to the security team within 24 hours.",
]
HANDBOOK_PAGES = [
    "Working hours are 9am to 6pm Monday to Friday. Remote work requires manager approval.",
    None,  # intentionally blank page
    "Dress code is business casual on all weekdays.",
]


@pytest.fixture
def leave_pdf():
    return make_pdf(LEAVE_PAGES)


@pytest.fixture
def it_pdf():
    return make_pdf(IT_PAGES)


@pytest.fixture
def handbook_pdf():
    return make_pdf(HANDBOOK_PAGES)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        llm_api_key="test-key",
        chunk_size=40, chunk_overlap=8,
        top_k=3, min_similarity=0.2,
        chroma_dir=tmp_path / "chroma", upload_dir=tmp_path / "uploads",
        registry_path=tmp_path / "documents.json",
        max_upload_mb=1,
    )


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def container(settings, embedder, llm):
    return Container.build(settings, embedder=embedder, llm=llm)


@pytest.fixture
def client(container):
    with TestClient(create_app(container=container)) as c:
        yield c


def upload(client, *named_files: tuple[str, bytes]):
    files = [("files", (name, data, "application/pdf")) for name, data in named_files]
    return client.post("/documents/upload", files=files)
