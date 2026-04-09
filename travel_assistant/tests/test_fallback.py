"""
Tests for handle_fallback node.

handle_fallback is the general chat handler — reached when:
  - intent confidence is too low
  - intent is INTENT_FALLBACK_GENERAL_TRAVEL
  - routing action is FALLBACK

Branches to cover:
  - Returns an AIMessage
  - Location present → city name in prompt
  - Location absent → "NOT AVAILABLE" in prompt  
  - nearby_context present → injected into prompt
  - nearby_context absent → "NOT AVAILABLE" in prompt
  - last_results present → injected into prompt
  - last_results absent → "No previous results" in prompt
  - user profile from store → injected into prompt
  - location_history_text → injected into prompt
  - user messages passed to LLM
  - does NOT tag last_results with handler name (unlike intent handlers)
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

LLM_PATH = "langchain_openai.ChatOpenAI.invoke"


# ── helpers ───────────────────────────────────────────────────────────────────

class FakeItem:
    def __init__(self, val):
        self.value = val


class FakeStore:
    def __init__(self, profile=None, travel_history=None):
        self.data = {}
        if profile:
            self.data[("user_profile", "test-user", "profile")] = profile
        if travel_history:
            self.data[("travel_history", "test-user", "history")] = travel_history

    def get(self, ns, key):
        k = tuple(ns) + (key,)
        return FakeItem(self.data[k]) if k in self.data else None

    def put(self, ns, key, value):
        k = tuple(ns) + (key,)
        self.data[k] = value


FAKE_CONFIG = {
    "configurable": {
        "user_id": "test-user",
        "thread_id": "test-thread",
        "location": None,
        "nearby_context": None,
        "connected_accounts": {},
    }
}

FAKE_TIME = {
    "local_time": "14:00",
    "day_of_week": "Monday",
    "hour": 14,
    "is_weekend": False
}


def make_state(**overrides):
    base = {
        "messages": [HumanMessage(content="hello")],
        "location": None,
        "nearby_context": None,
        "time_context": FAKE_TIME,
        "preferences": None,
        "last_results": None,
        "location_history_text": "No location history yet",
        "itinerary_context": {},
        "itinerary_messages": [],
        "itinerary_places": [],
        "party": None,
        "constraints": None,
        "connected_accounts": None,
        "safety_mode": "normal",
        "clarification_attempts": 0,
        "classification": None,
        "routing": None,
        "previous_intent": None,
    }
    base.update(overrides)
    return base


def fake_llm_response(content="Fallback response"):
    mock = MagicMock()
    mock.content = content
    return mock


# ═══════════════════════════════════════════════════════════════════════════════
# handle_fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleFallback:

    def test_returns_ai_message(self):
        """Basic output test — handler must return exactly one AIMessage."""
        from travel_assistant.chatbot import handle_fallback

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response("How can I help?")):
            result = handle_fallback(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "How can I help?"

    def test_does_not_tag_last_results(self):
        """Unlike intent handlers, handle_fallback does NOT write to
        last_results. This is intentional — fallback should not interfere
        with follow-up routing back to real intent handlers."""
        from travel_assistant.chatbot import handle_fallback

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response()):
            result = handle_fallback(state, FAKE_CONFIG, store=FakeStore())

        assert "last_results" not in result

    def test_location_city_in_prompt_when_location_given(self):
        """When location is present, city name should appear in system prompt."""
        from travel_assistant.chatbot import handle_fallback

        state = make_state(location={"city": "Amsterdam", "lat": 52.37, "lng": 4.90})
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_fallback(state, FAKE_CONFIG, store=FakeStore())

        assert "Amsterdam" in captured["system"]

    def test_not_available_in_prompt_when_no_location(self):
        """When location is None, prompt should contain NOT AVAILABLE."""
        from travel_assistant.chatbot import handle_fallback

        state = make_state(location=None)
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_fallback(state, FAKE_CONFIG, store=FakeStore())

        assert "NOT AVAILABLE" in captured["system"]

    def test_nearby_context_injected_when_present(self):
        """nearby_context string should appear in the system prompt."""
        from travel_assistant.chatbot import handle_fallback

        state = make_state(nearby_context="1. Cafe Nero — 150m\n2. Pret — 200m")
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_fallback(state, FAKE_CONFIG, store=FakeStore())

        assert "Cafe Nero" in captured["system"]

    def test_not_available_in_prompt_when_no_nearby_context(self):
        """When nearby_context is None, prompt should contain NOT AVAILABLE
        in the nearby places section."""
        from travel_assistant.chatbot import handle_fallback

        state = make_state(nearby_context=None)
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_fallback(state, FAKE_CONFIG, store=FakeStore())

        assert "NOT AVAILABLE" in captured["system"]

    def test_previous_results_injected_when_present(self):
        """When last_results exists, its content should appear in the prompt
        so the LLM can build on previous recommendations."""
        from travel_assistant.chatbot import handle_fallback

        prev = [{"handler": "INTENT_A_NEARBY_GENERIC", "response": "Here were the nearby cafes."}]
        state = make_state(last_results=prev)
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_fallback(state, FAKE_CONFIG, store=FakeStore())

        assert "nearby cafes" in captured["system"]

    def test_no_previous_results_fallback_text_in_prompt(self):
        """When last_results is None, prompt should contain
        'No previous results' placeholder."""
        from travel_assistant.chatbot import handle_fallback

        state = make_state(last_results=None)
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_fallback(state, FAKE_CONFIG, store=FakeStore())

        assert "No previous results" in captured["system"]

    def test_user_profile_injected_from_store(self):
        """When a user profile exists in store, name should appear in prompt."""
        from travel_assistant.chatbot import handle_fallback

        store = FakeStore(profile={
            "user_name": "Diana",
            "interests": ["art"],
            "dislikes": [],
            "additional_notes": None,
            "age": None,
            "location": None,
        })
        state = make_state()
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_fallback(state, FAKE_CONFIG, store=store)

        assert "Diana" in captured["system"]

    def test_location_history_injected_into_prompt(self):
        """location_history_text from state should appear in the prompt."""
        from travel_assistant.chatbot import handle_fallback

        state = make_state(
            location_history_text="2025-01-01 10:00 — Baker Street (51.52, -0.15)"
        )
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_fallback(state, FAKE_CONFIG, store=FakeStore())

        assert "Baker Street" in captured["system"]

    def test_user_messages_passed_to_llm(self):
        """The user's HumanMessage should be in the messages list
        sent to the LLM — not just the system prompt."""
        from travel_assistant.chatbot import handle_fallback

        user_msg = HumanMessage(content="tell me something interesting")
        state = make_state(messages=[user_msg])
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["messages"] = messages
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_fallback(state, FAKE_CONFIG, store=FakeStore())

        contents = [m.content for m in captured["messages"]]
        assert "tell me something interesting" in contents

    def test_no_crash_when_no_location_and_no_nearby(self):
        """Both location and nearby_context are None — handler
        should complete without any AttributeError or KeyError."""
        from travel_assistant.chatbot import handle_fallback

        state = make_state(location=None, nearby_context=None)

        with patch(LLM_PATH, return_value=fake_llm_response()):
            result = handle_fallback(state, FAKE_CONFIG, store=FakeStore())

        assert "messages" in result