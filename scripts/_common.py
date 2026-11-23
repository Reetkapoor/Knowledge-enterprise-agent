"""Shared bootstrap for the CLI scripts (run from the project root: python -m scripts.run_eval ...)."""
from __future__ import annotations

import logging

from backend.core.config import get_settings
from backend.core.container import Container


def build_container() -> Container:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if not settings.llm_configured:
        raise SystemExit("LLM_API_KEY is not set (see .env.example).")
    return Container.build(settings)
