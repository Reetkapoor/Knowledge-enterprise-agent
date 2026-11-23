"""Optional LLM-as-judge for answer correctness against the human reference answer.

Caveat (also in the README): the judge is an LLM, possibly from the same provider as the generator. Report it next to
the deterministic key-fact recall, and spot-check it by hand.
"""
from __future__ import annotations

from pydantic import BaseModel

from backend.services.llm_json import generate_structured
from backend.services.llm_service import LLMProvider

_SYSTEM = """You grade an answer against a human-written reference answer.
Score 1.0 if the answer states the same essential facts as the reference with no contradicting statement; 0.5 if it is
partly correct or incomplete but nothing in it contradicts the reference; 0.0 if it is wrong, contradicts the reference,
or gives an answer where the reference says the information is unavailable (and vice versa). Ignore style and citations.
Reply with one JSON object and nothing else: {"score": 0.0|0.5|1.0, "reason": "<one sentence>"}"""


class _Out(BaseModel):
    score: float
    reason: str = ""


def judge_answer(llm: LLMProvider, question: str, reference: str, answer: str) -> float:
    prompt = f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference or '(none: the documents do not contain the answer)'}\n\nANSWER: {answer}"
    score = generate_structured(llm, _SYSTEM, prompt, _Out, max_tokens=150).score
    return min(1.0, max(0.0, round(score * 2) / 2))
