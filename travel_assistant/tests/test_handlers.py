"""
Tests for all handler nodes:
  - handle_nearby_generic
  - handle_nearby_by_need
  - handle_food_dietary
  - handle_friends_based
  - handle_safety_practical
  - handle_itinerary
  - handle_urgent

Mocking strategy:
  patch("langchain_openai.ChatOpenAI.invoke", ...) patches the METHOD on the
  CLASS, bypassing Pydantic's instance __setattr__ guard entirely.
  patch.object(instance) and monkeypatch.setattr(instance) both fail because
  Pydantic blocks __setattr__ on instances — class-level patching avoids this.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

LLM_PATH = "langchain_openai.ChatOpenAI.invoke"


# ── shared helpers ────────────────────────────────────────────────────────────

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

FAKE_TIME = {"local_time": "14:00", "day_of_week": "Monday", "hour": 14, "is_weekend": False}
FAKE_PREFS = {"budget": "mid-range", "cuisine": "local", "vibe": "casual", "pace": "moderate"}


def make_state(**overrides):
    base = {
        "messages": [HumanMessage(content="test message")],
        "location": None,
        "nearby_context": None,
        "time_context": FAKE_TIME,
        "preferences": FAKE_PREFS,
        "last_results": None,
        "location_history_text": "No location history yet",
        "itinerary_context": {},
        "itinerary_messages": [],
        "itinerary_places": [],
        "party": None,
        "constraints": None,
        "connected_accounts": None,
        "safety_mode": None,
        "clarification_attempts": 0,
        "classification": None,
        "routing": None,
        "previous_intent": None,
    }
    base.update(overrides)
    return base


def fake_llm_response(content="LLM response"):
    mock = MagicMock()
    mock.content = content
    return mock


# ═══════════════════════════════════════════════════════════════════════════════
# handle_nearby_generic
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleNearbyGeneric:

    def test_returns_ai_message(self):
        from travel_assistant.chatbot import handle_nearby_generic

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response("Here are some cafes.")):
            result = handle_nearby_generic(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "Here are some cafes."

    def test_records_correct_handler_in_last_results(self):
        from travel_assistant.chatbot import handle_nearby_generic
        from travel_assistant.intents import Intent

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response()):
            result = handle_nearby_generic(state, FAKE_CONFIG, store=FakeStore())

        assert result["last_results"][0]["handler"] == Intent.INTENT_A_NEARBY_GENERIC.value

    def test_last_results_contains_response_text(self):
        from travel_assistant.chatbot import handle_nearby_generic

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response("Cafe list here")):
            result = handle_nearby_generic(state, FAKE_CONFIG, store=FakeStore())

        assert result["last_results"][0]["response"] == "Cafe list here"

    def test_location_context_not_available_when_no_location(self):
        from travel_assistant.chatbot import handle_nearby_generic

        state = make_state(location=None)
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_nearby_generic(state, FAKE_CONFIG, store=FakeStore())

        assert "NOT AVAILABLE" in captured["system"]

    def test_location_context_present_when_location_given(self):
        from travel_assistant.chatbot import handle_nearby_generic

        state = make_state(location={"city": "Tokyo", "lat": 35.68, "lng": 139.69})
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_nearby_generic(state, FAKE_CONFIG, store=FakeStore())

        assert "Tokyo" in captured["system"]

    def test_nearby_context_injected_into_prompt(self):
        from travel_assistant.chatbot import handle_nearby_generic

        state = make_state(nearby_context="1. Starbucks — 100m\n2. Costa — 200m")
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_nearby_generic(state, FAKE_CONFIG, store=FakeStore())

        assert "Starbucks" in captured["system"]

    def test_user_profile_injected_when_present(self):
        from travel_assistant.chatbot import handle_nearby_generic

        store = FakeStore(profile={
            "user_name": "Alex", "interests": ["hiking"],
            "dislikes": [], "additional_notes": None, "age": None, "location": None
        })
        state = make_state()
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_nearby_generic(state, FAKE_CONFIG, store=store)

        assert "Alex" in captured["system"]

    def test_user_messages_passed_to_llm(self):
        from travel_assistant.chatbot import handle_nearby_generic

        user_msg = HumanMessage(content="find me a pharmacy")
        state = make_state(messages=[user_msg])
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["messages"] = messages
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_nearby_generic(state, FAKE_CONFIG, store=FakeStore())

        contents = [m.content for m in captured["messages"]]
        assert "find me a pharmacy" in contents


# ═══════════════════════════════════════════════════════════════════════════════
# handle_nearby_by_need
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleNearbyByNeed:

    def test_returns_ai_message(self):
        from travel_assistant.chatbot import handle_nearby_by_need

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response("Try a park!")):
            result = handle_nearby_by_need(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1

    def test_records_correct_handler(self):
        from travel_assistant.chatbot import handle_nearby_by_need
        from travel_assistant.intents import Intent

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response()):
            result = handle_nearby_by_need(state, FAKE_CONFIG, store=FakeStore())

        assert result["last_results"][0]["handler"] == Intent.INTENT_B_NEARBY_BY_NEED.value

    def test_nearby_context_in_prompt(self):
        from travel_assistant.chatbot import handle_nearby_by_need

        state = make_state(nearby_context="1. Central Park — 300m")
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_nearby_by_need(state, FAKE_CONFIG, store=FakeStore())

        assert "Central Park" in captured["system"]

    def test_no_location_shows_not_available(self):
        from travel_assistant.chatbot import handle_nearby_by_need

        state = make_state(location=None)
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_nearby_by_need(state, FAKE_CONFIG, store=FakeStore())

        assert "NOT AVAILABLE" in captured["system"]


# ═══════════════════════════════════════════════════════════════════════════════
# handle_food_dietary
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleFoodDietary:

    def test_returns_ai_message(self):
        from travel_assistant.chatbot import handle_food_dietary

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response("Vegan options nearby.")):
            result = handle_food_dietary(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "Vegan options nearby."

    def test_records_correct_handler(self):
        from travel_assistant.chatbot import handle_food_dietary
        from travel_assistant.intents import Intent

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response()):
            result = handle_food_dietary(state, FAKE_CONFIG, store=FakeStore())

        assert result["last_results"][0]["handler"] == Intent.INTENT_D_FOOD_DIETARY.value

    def test_location_context_in_prompt(self):
        from travel_assistant.chatbot import handle_food_dietary

        state = make_state(location={"city": "Berlin", "lat": 52.52, "lng": 13.40})
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_food_dietary(state, FAKE_CONFIG, store=FakeStore())

        assert "Berlin" in captured["system"]

    def test_previous_results_passed_to_prompt(self):
        from travel_assistant.chatbot import handle_food_dietary

        prev = [{"handler": "INTENT_D_FOOD_DIETARY", "response": "Here were the halal options."}]
        state = make_state(last_results=prev)
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_food_dietary(state, FAKE_CONFIG, store=FakeStore())

        assert "halal options" in captured["system"]


# ═══════════════════════════════════════════════════════════════════════════════
# handle_friends_based
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleFriendsBased:

    def test_returns_ai_message(self):
        from travel_assistant.chatbot import handle_friends_based

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response("Your friends liked X.")):
            result = handle_friends_based(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1

    def test_records_correct_handler(self):
        from travel_assistant.chatbot import handle_friends_based
        from travel_assistant.intents import Intent

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response()):
            result = handle_friends_based(state, FAKE_CONFIG, store=FakeStore())

        assert result["last_results"][0]["handler"] == Intent.INTENT_E_FRIENDS_BASED.value

    def test_travel_history_injected_from_store(self):
        from travel_assistant.chatbot import handle_friends_based

        travel_history = [
            {"number": 1, "city": "Rome", "country": "Italy",
             "places_visited": ["Colosseum"], "time_of_visit": "2024-06", "hours_spent": 5}
        ]
        store = FakeStore(travel_history=travel_history)
        state = make_state()
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_friends_based(state, FAKE_CONFIG, store=store)

        assert "Rome" in captured["system"]

    def test_no_crash_when_no_location(self):
        from travel_assistant.chatbot import handle_friends_based

        state = make_state(location=None)
        with patch(LLM_PATH, return_value=fake_llm_response()):
            result = handle_friends_based(state, FAKE_CONFIG, store=FakeStore())

        assert "messages" in result


# ═══════════════════════════════════════════════════════════════════════════════
# handle_safety_practical
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleSafetyPractical:

    def test_returns_ai_message(self):
        from travel_assistant.chatbot import handle_safety_practical

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response("Stay safe!")):
            result = handle_safety_practical(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1

    def test_records_correct_handler(self):
        from travel_assistant.chatbot import handle_safety_practical
        from travel_assistant.intents import Intent

        state = make_state()
        with patch(LLM_PATH, return_value=fake_llm_response()):
            result = handle_safety_practical(state, FAKE_CONFIG, store=FakeStore())

        assert result["last_results"][0]["handler"] == Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP.value

    def test_no_crash_when_no_location(self):
        from travel_assistant.chatbot import handle_safety_practical

        state = make_state(location=None)
        with patch(LLM_PATH, return_value=fake_llm_response()):
            result = handle_safety_practical(state, FAKE_CONFIG, store=FakeStore())

        assert "messages" in result

    def test_previous_results_in_prompt(self):
        from travel_assistant.chatbot import handle_safety_practical

        prev = [{"handler": "INTENT_F", "response": "Avoid area X at night."}]
        state = make_state(last_results=prev)
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_safety_practical(state, FAKE_CONFIG, store=FakeStore())

        assert "Avoid area X" in captured["system"]


# ═══════════════════════════════════════════════════════════════════════════════
# handle_itinerary
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleItinerary:

    FULL_CTX = {
        "destination": "Paris",
        "current_location": "London",
        "num_days": 3,
        "party_size": 2,
        "transport_to": "train",
        "transport_within": "metro",
        "cuisine": "local",
        "interests": "history",
        "dest_lat": 48.85,
        "dest_lng": 2.35,
        "travel_distance_km": 340.0,
    }

    def test_returns_ai_message(self):
        from travel_assistant.chatbot import handle_itinerary

        state = make_state(itinerary_context=dict(self.FULL_CTX))
        with patch(LLM_PATH, return_value=fake_llm_response("Day 1 — Explore Montmartre...")):
            result = handle_itinerary(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1

    def test_records_correct_handler(self):
        from travel_assistant.chatbot import handle_itinerary
        from travel_assistant.intents import Intent

        state = make_state(itinerary_context=dict(self.FULL_CTX))
        with patch(LLM_PATH, return_value=fake_llm_response()):
            result = handle_itinerary(state, FAKE_CONFIG, store=FakeStore())

        assert result["last_results"][0]["handler"] == Intent.INTENT_C_ITINERARY.value

    def test_destination_in_prompt(self):
        from travel_assistant.chatbot import handle_itinerary

        state = make_state(itinerary_context=dict(self.FULL_CTX))
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_itinerary(state, FAKE_CONFIG, store=FakeStore())

        assert "Paris" in captured["system"]

    def test_places_block_injected_when_places_available(self):
        from travel_assistant.chatbot import handle_itinerary

        places = [{"name": "Eiffel Tower", "type": "attraction", "distance": 200,
                   "cuisine": "", "opening_hours": ""}]
        state = make_state(itinerary_context=dict(self.FULL_CTX), itinerary_places=places)
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_itinerary(state, FAKE_CONFIG, store=FakeStore())

        assert "Eiffel Tower" in captured["system"]

    def test_fallback_message_when_no_places(self):
        from travel_assistant.chatbot import handle_itinerary

        state = make_state(itinerary_context=dict(self.FULL_CTX), itinerary_places=[])
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_itinerary(state, FAKE_CONFIG, store=FakeStore())

        assert "general knowledge" in captured["system"].lower() or "no place data" in captured["system"].lower()

    def test_travel_distance_in_prompt(self):
        from travel_assistant.chatbot import handle_itinerary

        state = make_state(itinerary_context=dict(self.FULL_CTX))
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_itinerary(state, FAKE_CONFIG, store=FakeStore())

        assert "340" in captured["system"]

    def test_conversation_context_appended_when_messages_exist(self):
        from travel_assistant.chatbot import handle_itinerary

        msgs = [
            HumanMessage(content="I want 3 days in Paris"),
            AIMessage(content="Great! How many people?"),
        ]
        state = make_state(itinerary_context=dict(self.FULL_CTX))
        state["itinerary_messages"] = msgs
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["system"] = messages[0].content
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_itinerary(state, FAKE_CONFIG, store=FakeStore())

        assert "CONVERSATION CONTEXT" in captured["system"]
        assert "3 days in Paris" in captured["system"]

    def test_itinerary_messages_preserved_in_return(self):
        from travel_assistant.chatbot import handle_itinerary

        msgs = [HumanMessage(content="Paris trip"), AIMessage(content="How many days?")]
        state = make_state(itinerary_context=dict(self.FULL_CTX))
        state["itinerary_messages"] = msgs

        with patch(LLM_PATH, return_value=fake_llm_response()):
            result = handle_itinerary(state, FAKE_CONFIG, store=FakeStore())

        assert result["itinerary_messages"] == msgs

    def test_only_last_3_user_messages_sent_to_llm(self):
        from travel_assistant.chatbot import handle_itinerary

        many_messages = [HumanMessage(content=f"msg {i}") for i in range(10)]
        state = make_state(itinerary_context=dict(self.FULL_CTX), messages=many_messages)
        captured = {}

        def capture(self_llm, messages, **kwargs):
            captured["messages"] = messages
            return fake_llm_response()

        with patch(LLM_PATH, capture):
            handle_itinerary(state, FAKE_CONFIG, store=FakeStore())

        non_system = [m for m in captured["messages"] if not isinstance(m, SystemMessage)]
        assert len(non_system) <= 3


# ═══════════════════════════════════════════════════════════════════════════════
# handle_urgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleUrgent:

    def test_appends_emergency_message(self):
        from travel_assistant.chatbot import handle_urgent

        state = make_state()
        result = handle_urgent(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert any("medical" in m.content.lower() or "🚨" in m.content for m in ai_msgs)

    def test_does_not_call_llm(self):
        from travel_assistant.chatbot import handle_urgent

        state = make_state()
        with patch(LLM_PATH) as mock_invoke:
            handle_urgent(state, FAKE_CONFIG, store=FakeStore())
            mock_invoke.assert_not_called()