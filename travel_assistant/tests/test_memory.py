"""
Tests for write_memory node.

Patches:
  - trustcall_extractor at class level (RunnableSequence)
    because Pydantic blocks instance-level patching
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

TRUSTCALL_PATH = "langchain_core.runnables.base.RunnableSequence.invoke"


# ── helpers ───────────────────────────────────────────────────────────────────

class FakeItem:
    def __init__(self, val):
        self.value = val


class FakeStore:
    def __init__(self):
        self.data = {}

    def get(self, ns, key):
        k = tuple(ns) + (key,)
        return FakeItem(self.data[k]) if k in self.data else None

    def put(self, ns, key, value):
        k = tuple(ns) + (key,)
        self.data[k] = value


def make_state(**overrides):
    base = {
        "messages": [HumanMessage(content="I love hiking")],
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


FAKE_CONFIG = {
    "configurable": {
        "user_id": "test-user",
        "thread_id": "test-thread",
    }
}

NO_USER_CONFIG = {
    "configurable": {
        "user_id": None,
        "thread_id": "test-thread",
    }
}


def fake_trustcall_response(user_name="TestUser"):
    """Returns a minimal trustcall result with one profile response."""
    profile = MagicMock()
    profile.model_dump.return_value = {
        "user_name": user_name,
        "age": None,
        "location": None,
        "interests": [],
        "dislikes": [],
        "additional_notes": None,
    }
    return {"responses": [profile]}


def empty_trustcall_response():
    """Returns trustcall result with no responses — simulates no extractable data."""
    return {"responses": []}


# ═══════════════════════════════════════════════════════════════════════════════
# write_memory
# ═══════════════════════════════════════════════════════════════════════════════

class TestWriteMemory:

    def test_returns_state_when_no_user_id(self):
        """If user_id is None, write_memory should return state immediately
        without touching the store or calling trustcall."""
        from travel_assistant.chatbot import write_memory

        store = FakeStore()
        state = make_state()

        with patch(TRUSTCALL_PATH) as mock_trustcall:
            result = write_memory(state, NO_USER_CONFIG, store=store)
            mock_trustcall.assert_not_called()

        # Store should be completely untouched
        assert store.data == {}

    def test_calls_store_put_with_extracted_profile(self):
        """Happy path — trustcall extracts a profile → store.put() is called."""
        from travel_assistant.chatbot import write_memory

        store = FakeStore()
        state = make_state()

        with patch(TRUSTCALL_PATH, return_value=fake_trustcall_response("Alice")):
            write_memory(state, FAKE_CONFIG, store=store)

        # Store should now have the profile saved
        key = (("user_profile", "test-user"), "profile")
        saved = store.get(("user_profile", "test-user"), "profile")
        assert saved is not None
        assert saved.value["user_name"] == "Alice"

    def test_skips_store_put_when_no_responses(self):
        """If trustcall returns empty responses list, store.put() must NOT be called.
        This is the guard added to prevent overwriting existing profile with nothing."""
        from travel_assistant.chatbot import write_memory

        store = FakeStore()
        state = make_state()

        with patch(TRUSTCALL_PATH, return_value=empty_trustcall_response()):
            write_memory(state, FAKE_CONFIG, store=store)

        # Store should be untouched — no profile written
        saved = store.get(("user_profile", "test-user"), "profile")
        assert saved is None

    def test_passes_existing_profile_to_trustcall(self):
        """If a profile already exists in store, it should be passed to trustcall
        as 'existing' so it can update rather than overwrite."""
        from travel_assistant.chatbot import write_memory

        store = FakeStore()
        # Pre-seed with existing profile
        store.put(
            ("user_profile", "test-user"),
            "profile",
            {"user_name": "Bob", "interests": ["hiking"], "dislikes": [],
             "additional_notes": None, "age": None, "location": None}
        )
        state = make_state()
        captured = {}

        def capture_trustcall(self_seq, input_data, **kwargs):
            captured["existing"] = input_data.get("existing")
            return fake_trustcall_response("Bob")

        with patch(TRUSTCALL_PATH, capture_trustcall):
            write_memory(state, FAKE_CONFIG, store=store)

        # existing profile should have been passed in
        assert captured["existing"] is not None
        assert captured["existing"]["UserProfile"]["user_name"] == "Bob"

    def test_passes_none_existing_when_no_prior_profile(self):
        """If no profile in store yet, 'existing' passed to trustcall should be None."""
        from travel_assistant.chatbot import write_memory

        store = FakeStore()  # empty store
        state = make_state()
        captured = {}

        def capture_trustcall(self_seq, input_data, **kwargs):
            captured["existing"] = input_data.get("existing")
            return fake_trustcall_response()

        with patch(TRUSTCALL_PATH, capture_trustcall):
            write_memory(state, FAKE_CONFIG, store=store)

        assert captured["existing"] is None

    def test_conversation_messages_passed_to_trustcall(self):
        """The full conversation messages should be passed to trustcall
        so it can extract profile info from what the user said."""
        from travel_assistant.chatbot import write_memory

        store = FakeStore()
        state = make_state(messages=[
            HumanMessage(content="My name is Carol and I love beaches")
        ])
        captured = {}

        def capture_trustcall(self_seq, input_data, **kwargs):
            captured["messages"] = input_data.get("messages", [])
            return fake_trustcall_response("Carol")

        with patch(TRUSTCALL_PATH, capture_trustcall):
            write_memory(state, FAKE_CONFIG, store=store)

        # User message should be in the messages passed to trustcall
        contents = [m.content for m in captured["messages"] if hasattr(m, "content")]
        assert any("Carol" in c for c in contents)

    def test_returns_state_unchanged(self):
        """write_memory should return the same state object it received,
        not a modified copy."""
        from travel_assistant.chatbot import write_memory

        store = FakeStore()
        state = make_state()

        with patch(TRUSTCALL_PATH, return_value=fake_trustcall_response()):
            result = write_memory(state, FAKE_CONFIG, store=store)

        assert result is state