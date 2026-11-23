from __future__ import annotations

import pytest

from backend.schemas.agentic import QueryType
from backend.services.llm_service import LLMError
from backend.services.query_router import QueryRouter
from tests.helpers import ScriptedLLM


def route(resp, max_sub=3, q="Compare leave for permanent and contractual staff"):
    return QueryRouter(ScriptedLLM(router=resp), max_sub).route(q)


def test_simple_has_no_sub_questions():
    plan = route({"query_type": "SIMPLE", "sub_questions": ["ignored"], "reasoning": "one fact"})
    assert plan.query_type == QueryType.SIMPLE and plan.sub_questions == []


def test_complex_decomposition_is_cleaned_and_capped():
    plan = route({"query_type": "COMPLEX", "reasoning": "comparison", "sub_questions": [
        "Leave policy for permanent employees?", "leave policy for permanent employees?",
        "Leave policy for contractual employees?", "Probation rules?", "Notice period?"]}, max_sub=3)
    assert plan.query_type == QueryType.COMPLEX
    assert plan.sub_questions == ["Leave policy for permanent employees?", "Leave policy for contractual employees?", "Probation rules?"]


def test_complex_without_usable_decomposition_is_downgraded_to_simple():
    plan = route({"query_type": "COMPLEX", "sub_questions": ["Only one sub question here?"]})
    assert plan.query_type == QueryType.SIMPLE


def test_conversational_gets_reply_without_sub_questions():
    plan = route({"query_type": "CONVERSATIONAL", "conversational_reply": "Hello!"})
    assert plan.query_type == QueryType.CONVERSATIONAL and plan.conversational_reply == "Hello!"


def test_garbage_output_falls_back_to_simple_with_error():
    plan = route("I think this is complex, honestly")
    assert plan.query_type == QueryType.SIMPLE and plan.error


def test_missing_api_key_is_not_swallowed():
    def boom(_):
        raise LLMError("no key", not_configured=True)
    with pytest.raises(LLMError):
        route(boom)
