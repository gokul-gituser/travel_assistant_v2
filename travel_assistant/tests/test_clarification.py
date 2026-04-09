"""
Tests for:
  - handle_clarification
  - get_clarification_question

Branches covered:
  handle_clarification:
    - attempts=0 → asks clarification question
    - attempts=1 → asks clarification question
    - attempts=2 → max attempts reached, shows fallback message
    - attempts counter incremented correctly each call
    - returned state contains AI message

  get_clarification_question:
    - no classification in state → returns default question
    - classification present → calls clarification_agent
    - agent returns messages → uses last message content
    - agent returns empty messages → returns default question
    - agent raises exception → returns fallback question
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage


# ── helpers ───────────────────────────────────────────────────────────────────

FAKE_CONFIG = {
    "configurable": {
        "user_id": "test-user",
        "thread_id": "test-thread",
        "location": None,
        "nearby_context": None,
        "connected_accounts": {},
    }
}

CLARIFICATION_AGENT_PATH = "travel_assistant.chatbot.clarification_agent.invoke"


class FakeStore:
    def get(self, ns, key): return None
    def put(self, ns, key, value): pass


def make_state(**overrides):
    base = {
        "messages": [HumanMessage(content="something ambiguous")],
        "location": None,
        "nearby_context": None,
        "time_context": None,
        "party": None,
        "preferences": None,
        "constraints": None,
        "connected_accounts": None,
        "safety_mode": "normal",
        "last_results": None,
        "location_history_text": None,
        "previous_intent": None,
        "itinerary_context": {},
        "itinerary_places": [],
        "itinerary_messages": [],
        "classification": None,
        "routing": None,
        "clarification_attempts": 0,
    }
    base.update(overrides)
    return base


def make_classification_with_scores(top_intent="INTENT_A_NEARBY_GENERIC", top_score=0.5):
    """Build a classification dict with competing intent scores."""
    remainder = round((1.0 - top_score) / 7, 4)
    scores = {
        "INTENT_A_NEARBY_GENERIC": remainder,
        "INTENT_B_NEARBY_BY_NEED": remainder,
        "INTENT_C_ITINERARY": remainder,
        "INTENT_D_FOOD_DIETARY": remainder,
        "INTENT_E_FRIENDS_BASED": remainder,
        "INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP": remainder,
        "INTENT_G_URGENT_HEALTH": remainder,
        "INTENT_FALLBACK_GENERAL_TRAVEL": remainder,
    }
    scores[top_intent] = top_score
    return {
        "primary_intent": top_intent,
        "confidence": top_score,
        "intent_scores": scores,
        "needs_clarification": True,
        "safety_override": False,
    }


def fake_agent_response(question="Are you looking for a place nearby or planning a trip?"):
    """Simulates clarification_agent.invoke() return value."""
    msg = MagicMock()
    msg.content = question
    return {"messages": [msg]}


# ═══════════════════════════════════════════════════════════════════════════════
# handle_clarification
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleClarification:

    def test_appends_ai_message_on_first_attempt(self):
        """First clarification attempt should produce an AI message."""
        from travel_assistant.chatbot import handle_clarification

        state = make_state(clarification_attempts=0)

        with patch(CLARIFICATION_AGENT_PATH, return_value=fake_agent_response("What do you mean?")):
            result = handle_clarification(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1

    def test_increments_attempts_counter(self):
        """clarification_attempts must be incremented by 1 each call."""
        from travel_assistant.chatbot import handle_clarification

        state = make_state(clarification_attempts=0)

        with patch(CLARIFICATION_AGENT_PATH, return_value=fake_agent_response()):
            result = handle_clarification(state, FAKE_CONFIG, store=FakeStore())

        assert result["clarification_attempts"] == 1

    def test_increments_from_nonzero(self):
        """Counter increments correctly from any starting value."""
        from travel_assistant.chatbot import handle_clarification

        state = make_state(clarification_attempts=1)

        with patch(CLARIFICATION_AGENT_PATH, return_value=fake_agent_response()):
            result = handle_clarification(state, FAKE_CONFIG, store=FakeStore())

        assert result["clarification_attempts"] == 2

    def test_shows_fallback_message_at_max_attempts(self):
        """At attempts=2, should show a fixed fallback message instead of
        calling the clarification agent."""
        from travel_assistant.chatbot import handle_clarification

        state = make_state(clarification_attempts=2)

        with patch(CLARIFICATION_AGENT_PATH) as mock_agent:
            result = handle_clarification(state, FAKE_CONFIG, store=FakeStore())
            # Agent should NOT be called at max attempts
            mock_agent.assert_not_called()

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        # Should contain the hardcoded fallback text
        assert "trouble" in ai_msgs[0].content.lower() or \
               "general" in ai_msgs[0].content.lower() or \
               "understand" in ai_msgs[0].content.lower()

    def test_does_not_call_agent_at_max_attempts(self):
        """At max attempts, clarification_agent must never be invoked —
        the fallback message is hardcoded."""
        from travel_assistant.chatbot import handle_clarification

        state = make_state(clarification_attempts=2)

        with patch(CLARIFICATION_AGENT_PATH) as mock_agent:
            handle_clarification(state, FAKE_CONFIG, store=FakeStore())
            mock_agent.assert_not_called()

    def test_calls_agent_before_max_attempts(self):
        """Below max attempts, clarification_agent should be called."""
        from travel_assistant.chatbot import handle_clarification

        state = make_state(
            clarification_attempts=1,
            classification=make_classification_with_scores()
        )

        with patch(CLARIFICATION_AGENT_PATH, return_value=fake_agent_response()) as mock_agent:
            handle_clarification(state, FAKE_CONFIG, store=FakeStore())
            mock_agent.assert_called_once()

    def test_question_from_agent_used_as_message_content(self):
        """The question returned by the clarification agent should be
        the content of the AI message appended to state."""
        from travel_assistant.chatbot import handle_clarification

        question = "Are you looking for restaurants or activities?"
        state = make_state(
            clarification_attempts=0,
            classification=make_classification_with_scores()
        )

        with patch(CLARIFICATION_AGENT_PATH, return_value=fake_agent_response(question)):
            result = handle_clarification(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert ai_msgs[0].content == question

    def test_returns_state(self):
        """handle_clarification should return the state object."""
        from travel_assistant.chatbot import handle_clarification

        state = make_state()

        with patch(CLARIFICATION_AGENT_PATH, return_value=fake_agent_response()):
            result = handle_clarification(state, FAKE_CONFIG, store=FakeStore())

        assert result is state


# ═══════════════════════════════════════════════════════════════════════════════
# get_clarification_question
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetClarificationQuestion:

    def test_returns_default_when_no_classification(self):
        """If classification is None in state, should return the
        default fallback question without calling the agent."""
        from travel_assistant.chatbot import get_clarification_question

        state = make_state(classification=None)

        with patch(CLARIFICATION_AGENT_PATH) as mock_agent:
            result = get_clarification_question(state, FAKE_CONFIG, store=FakeStore())
            mock_agent.assert_not_called()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_calls_agent_when_classification_present(self):
        """When classification exists, clarification_agent should be called."""
        from travel_assistant.chatbot import get_clarification_question

        state = make_state(classification=make_classification_with_scores())

        with patch(CLARIFICATION_AGENT_PATH, return_value=fake_agent_response()) as mock_agent:
            get_clarification_question(state, FAKE_CONFIG, store=FakeStore())
            mock_agent.assert_called_once()

    def test_returns_agent_question(self):
        """The last message from the agent response should be returned."""
        from travel_assistant.chatbot import get_clarification_question

        question = "Do you want nearby places or a full itinerary?"
        state = make_state(classification=make_classification_with_scores())

        with patch(CLARIFICATION_AGENT_PATH, return_value=fake_agent_response(question)):
            result = get_clarification_question(state, FAKE_CONFIG, store=FakeStore())

        assert result == question

    def test_returns_default_when_agent_returns_empty_messages(self):
        """If agent returns empty messages list, should fall back to
        the default question string."""
        from travel_assistant.chatbot import get_clarification_question

        state = make_state(classification=make_classification_with_scores())

        with patch(CLARIFICATION_AGENT_PATH, return_value={"messages": []}):
            result = get_clarification_question(state, FAKE_CONFIG, store=FakeStore())

        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_fallback_when_agent_raises_exception(self):
        """If clarification_agent throws any exception, should catch it
        and return a fallback string — never crash."""
        from travel_assistant.chatbot import get_clarification_question

        state = make_state(classification=make_classification_with_scores())

        with patch(CLARIFICATION_AGENT_PATH, side_effect=Exception("Agent failed")):
            result = get_clarification_question(state, FAKE_CONFIG, store=FakeStore())

        assert isinstance(result, str)
        assert len(result) > 0

    def test_top_intents_included_in_agent_prompt(self):
        from travel_assistant.chatbot import get_clarification_question

        classification = make_classification_with_scores(
            top_intent="INTENT_A_NEARBY_GENERIC",
            top_score=0.5
        )
        state = make_state(classification=classification)
        captured = {}

        def capture_agent(input_data, **kwargs):  # ← removed self_agent
            captured["input"] = input_data
            return fake_agent_response()

        with patch(CLARIFICATION_AGENT_PATH, capture_agent):
            get_clarification_question(state, FAKE_CONFIG, store=FakeStore())

        messages = captured["input"].get("messages", [])
        assert len(messages) > 0
        content = messages[0]["content"]  # these are dicts with "role"/"content"
        assert "INTENT" in content


    def test_user_message_included_in_agent_prompt(self):
        from travel_assistant.chatbot import get_clarification_question

        state = make_state(
            messages=[HumanMessage(content="I want something fun")],
            classification=make_classification_with_scores()
        )
        captured = {}

        def capture_agent(input_data, **kwargs):  # ← removed self_agent
            captured["input"] = input_data
            return fake_agent_response()

        with patch(CLARIFICATION_AGENT_PATH, capture_agent):
            get_clarification_question(state, FAKE_CONFIG, store=FakeStore())

        messages = captured["input"].get("messages", [])
        content = messages[0]["content"]
        assert "something fun" in content