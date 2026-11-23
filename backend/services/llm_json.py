"""Structured LLM output: ask for JSON, parse it, validate it against a pydantic model.

Used by the router, retrieval evaluator, query refiner and claim extractor. One repair retry is
made when the output is not valid; after that StructuredOutputError is raised so callers can fall
back to a safe default instead of crashing the pipeline.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from backend.services.llm_service import LLMProvider

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    pass


def extract_json_object(text: str) -> dict:
    """Pull the first balanced {...} object out of an LLM reply (tolerates code fences / prose)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    raise StructuredOutputError("no JSON object found in LLM output")


def generate_structured(
    llm: LLMProvider, system: str, user: str, model: type[T], *, max_tokens: int = 600, retries: int = 1
) -> T:
    last_err = "unknown"
    prompt = user
    for attempt in range(retries + 1):
        raw = llm.generate(system, prompt, max_tokens=max_tokens)
        try:
            return model.model_validate(extract_json_object(raw))
        except (StructuredOutputError, ValidationError) as exc:
            last_err = str(exc).splitlines()[0][:200]
            logger.warning("Structured output invalid (attempt %d): %s", attempt + 1, last_err)
            prompt = (
                f"{user}\n\nYour previous reply was not valid ({last_err}). "
                "Reply again with ONLY the JSON object, matching the requested fields exactly."
            )
    raise StructuredOutputError(last_err)
