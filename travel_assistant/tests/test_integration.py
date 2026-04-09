"""
Layer 4 — Integration smoke tests for run_travel_assistant().

Pydantic blocks setattr on instances — both ChatOpenAI and trustcall_extractor
(a RunnableSequence) reject monkeypatch and patch.object.
Fix: patch at the CLASS level, which bypasses Pydantic's __setattr__ entirely.

  LLM:          patch("langchain_openai.ChatOpenAI.invoke")
  trustcall:    patch("langchain_core.runnables.base.RunnableSequence.invoke")

The trustcall patch is broad (affects all RunnableSequences) but is scoped
inside a with-block so it only applies for the duration of the test.
"""

import pytest
from unittest.mock import MagicMock, patch

LLM_PATH        = "langchain_openai.ChatOpenAI.invoke"
TRUSTCALL_PATH  = "langchain_core.runnables.base.RunnableSequence.invoke"


def fake_llm_response(content="Smoke test response."):
    mock = MagicMock()
    mock.content = content
    return mock


def fake_trustcall_response():
    """Returns a minimal trustcall result so write_memory doesn't call OpenAI."""
    profile = MagicMock()
    profile.model_dump.return_value = {"user_name": "Unknown", "interests": []}
    return {"responses": [profile]}


def make_fake_classifier(top_intent: str, score: float = 0.9):
    remainder = round((1.0 - score) / 7, 4)
    all_scores = {
        "INTENT_A_NEARBY_GENERIC": remainder,
        "INTENT_B_NEARBY_BY_NEED": remainder,
        "INTENT_C_ITINERARY": remainder,
        "INTENT_D_FOOD_DIETARY": remainder,
        "INTENT_E_FRIENDS_BASED": remainder,
        "INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP": remainder,
        "INTENT_G_URGENT_HEALTH": remainder,
        "INTENT_FALLBACK_GENERAL_TRAVEL": remainder,
    }
    all_scores[top_intent] = score

    def _classifier(*args, **kwargs):
        class FakeResponse:
            pass
        r = FakeResponse()
        r.all_scores = all_scores
        return {"structured_response": r}

    return _classifier


# ── smoke test 1: nearby intent ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_travel_assistant_returns_string(monkeypatch):
    """
    Full pipeline smoke test with INTENT_A_NEARBY_GENERIC.
    Asserts: returns a non-empty string. Nothing about content.
    """
    from travel_assistant.chatbot import run_travel_assistant

    monkeypatch.setattr(
        "travel_assistant.chatbot.classifier_agent.invoke",
        make_fake_classifier("INTENT_A_NEARBY_GENERIC")
    )

    with patch(LLM_PATH, return_value=fake_llm_response("Here are places near you.")), \
         patch(TRUSTCALL_PATH, return_value=fake_trustcall_response()):

        result = await run_travel_assistant(
            user_id="smoke-user-1",
            text="find cafes near me",
            location={"lat": 10.0, "lng": 20.0},
            thread_id="smoke-thread-1",
        )

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


# ── smoke test 2: itinerary intent ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_travel_assistant_itinerary_returns_string(monkeypatch):
    """
    Full pipeline smoke test with INTENT_C_ITINERARY.
    _extract_params is also mocked since collect_itinerary_context calls it.
    Asserts: returns a non-empty string. Nothing about content.
    """
    from travel_assistant.chatbot import run_travel_assistant

    monkeypatch.setattr(
        "travel_assistant.chatbot.classifier_agent.invoke",
        make_fake_classifier("INTENT_C_ITINERARY", score=0.95)
    )

    monkeypatch.setattr(
        "travel_assistant.chatbot._extract_params",
        lambda llm, message: {}
    )

    with patch(LLM_PATH, return_value=fake_llm_response("Where are you planning to travel?")), \
         patch(TRUSTCALL_PATH, return_value=fake_trustcall_response()):

        result = await run_travel_assistant(
            user_id="smoke-user-2",
            text="plan a trip",
            thread_id="smoke-thread-2",
        )

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0