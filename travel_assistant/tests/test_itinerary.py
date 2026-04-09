"""
Tests for the itinerary flow:
  - collect_itinerary_context
  - should_proceed_to_enrichment
  - enrich_itinerary_data
"""

import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage


# ── shared helpers ────────────────────────────────────────────────────────────

def make_state(itinerary_context=None, location=None, messages=None):
    """Minimal GraphState dict for itinerary tests."""
    return {
        "messages": messages or [HumanMessage(content="plan a trip")],
        "itinerary_context": itinerary_context or {},
        "itinerary_messages": [],
        "location": location,
        "nearby_context": None,
        "time_context": None,
        "preferences": None,
        "last_results": None,
        "location_history_text": None,
        "classification": None,
        "routing": None,
        "previous_intent": None,
        "party": None,
        "constraints": None,
        "connected_accounts": None,
        "safety_mode": None,
        "clarification_attempts": 0,
        "itinerary_places": [],
    }


FAKE_CONFIG = {
    "configurable": {
        "user_id": "test-user",
        "thread_id": "test-thread",
        "location": None,
        "nearby_context": None,
        "connected_accounts": {},
    }
}


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


FULL_CTX = {
    "destination": "Paris",
    "current_location": "London",
    "num_days": 3,
    "party_size": 2,
    "transport_to": "train",
    "transport_within": "metro",
    "cuisine": "local",
    "interests": "history",
}


# ═══════════════════════════════════════════════════════════════════════════════
# collect_itinerary_context
# ═══════════════════════════════════════════════════════════════════════════════

class TestCollectItineraryContext:

    def test_asks_for_destination_when_empty(self):
        """With no context at all, should ask for destination first."""
        from travel_assistant.chatbot import collect_itinerary_context

        state = make_state(messages=[HumanMessage(content="I want to plan a trip")])

        with patch("travel_assistant.chatbot._extract_params", return_value={}):
            result = collect_itinerary_context(state, FAKE_CONFIG, store=FakeStore())

        # Should have returned an AI message asking about destination
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        assert "travel" in ai_msgs[0].content.lower() or "destination" in ai_msgs[0].content.lower() or "planning" in ai_msgs[0].content.lower()

    def test_merges_extracted_params_into_context(self):
        """Extracted params should be merged into itinerary_context."""
        from travel_assistant.chatbot import collect_itinerary_context

        state = make_state(messages=[HumanMessage(content="I want to go to Tokyo for 5 days")])

        with patch("travel_assistant.chatbot._extract_params", return_value={"destination": "Tokyo", "num_days": 5}):
            result = collect_itinerary_context(state, FAKE_CONFIG, store=FakeStore())

        ctx = result["itinerary_context"]
        assert ctx["destination"] == "Tokyo"
        assert ctx["num_days"] == 5

    def test_does_not_overwrite_existing_context(self):
        """Already confirmed fields must not be overwritten by new extraction."""
        from travel_assistant.chatbot import collect_itinerary_context

        existing = {"destination": "Paris", "num_days": 3}
        state = make_state(
            itinerary_context=existing,
            messages=[HumanMessage(content="actually go to Rome for 7 days")]
        )

        # LLM extracts new values — should NOT overwrite confirmed ones
        with patch("travel_assistant.chatbot._extract_params", return_value={"destination": "Rome", "num_days": 7}):
            result = collect_itinerary_context(state, FAKE_CONFIG, store=FakeStore())

        ctx = result["itinerary_context"]
        assert ctx["destination"] == "Paris"   # unchanged
        assert ctx["num_days"] == 3            # unchanged

    def test_asks_fields_in_order(self):
        """After destination is set, it should ask for current_location next."""
        from travel_assistant.chatbot import collect_itinerary_context, COLLECTION_ORDER

        # Destination already known, nothing else
        state = make_state(
            itinerary_context={"destination": "Paris"},
            messages=[HumanMessage(content="Paris")]
        )

        with patch("travel_assistant.chatbot._extract_params", return_value={}):
            result = collect_itinerary_context(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        # Second field in COLLECTION_ORDER is current_location
        assert "current_location" == COLLECTION_ORDER[1]
        assert "traveling from" in ai_msgs[0].content.lower() or "current location" in ai_msgs[0].content.lower()

    def test_autofills_current_location_from_device(self):
        """If device location is available, current_location should be auto-filled."""
        from travel_assistant.chatbot import collect_itinerary_context

        state = make_state(
            itinerary_context={"destination": "Tokyo"},
            location={"lat": 51.5, "lng": -0.1, "city": "London"},
            messages=[HumanMessage(content="Tokyo")]
        )

        with patch("travel_assistant.chatbot._extract_params", return_value={}):
            result = collect_itinerary_context(state, FAKE_CONFIG, store=FakeStore())

        ctx = result["itinerary_context"]
        assert ctx["current_location"] == "London"

    def test_autofill_uses_coords_when_no_city(self):
        """If location has no city, auto-fill should use lat/lng string."""
        from travel_assistant.chatbot import collect_itinerary_context

        state = make_state(
            itinerary_context={"destination": "Tokyo"},
            location={"lat": 51.5, "lng": -0.1},  # no 'city' key
            messages=[HumanMessage(content="Tokyo")]
        )

        with patch("travel_assistant.chatbot._extract_params", return_value={}):
            result = collect_itinerary_context(state, FAKE_CONFIG, store=FakeStore())

        ctx = result["itinerary_context"]
        assert "51.5" in ctx["current_location"]
        assert "-0.1" in ctx["current_location"]

    def test_proceeds_when_all_fields_collected(self):
        """When all 8 fields are present, should NOT return an AI question."""
        from travel_assistant.chatbot import collect_itinerary_context

        state = make_state(
            itinerary_context=dict(FULL_CTX),
            messages=[HumanMessage(content="interests: history")]
        )

        with patch("travel_assistant.chatbot._extract_params", return_value={}):
            result = collect_itinerary_context(state, FAKE_CONFIG, store=FakeStore())

        # No AI question message — just state passthrough
        ai_msgs = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 0
        assert result["itinerary_context"] == FULL_CTX

    def test_stores_message_history(self):
        """User messages should accumulate in itinerary_messages."""
        from travel_assistant.chatbot import collect_itinerary_context

        state = make_state(messages=[HumanMessage(content="I want to go to Berlin")])

        with patch("travel_assistant.chatbot._extract_params", return_value={"destination": "Berlin"}):
            result = collect_itinerary_context(state, FAKE_CONFIG, store=FakeStore())

        # Should contain the user message + AI reply
        assert len(result["itinerary_messages"]) == 2
        assert result["itinerary_messages"][0].content == "I want to go to Berlin"

    def test_message_history_rolling_window(self):
        """Message history should be capped at 10 entries."""
        from travel_assistant.chatbot import collect_itinerary_context

        # Pre-fill 10 messages (the maximum)
        old_messages = [HumanMessage(content=f"msg {i}") for i in range(5)]
        old_messages += [AIMessage(content=f"reply {i}") for i in range(5)]

        state = make_state(messages=[HumanMessage(content="Berlin")])
        state["itinerary_messages"] = old_messages  # already at limit

        with patch("travel_assistant.chatbot._extract_params", return_value={"destination": "Berlin"}):
            result = collect_itinerary_context(state, FAKE_CONFIG, store=FakeStore())

        assert len(result["itinerary_messages"]) <= 10

    def test_formats_transport_question_with_destination(self):
        """The transport_to question should include the destination name."""
        from travel_assistant.chatbot import collect_itinerary_context

        # Fill everything up to transport_to
        ctx = {
            "destination": "Paris",
            "current_location": "London",
            "num_days": 3,
            "party_size": 2,
        }
        state = make_state(
            itinerary_context=ctx,
            messages=[HumanMessage(content="2 people")]
        )

        with patch("travel_assistant.chatbot._extract_params", return_value={}):
            result = collect_itinerary_context(state, FAKE_CONFIG, store=FakeStore())

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        assert "Paris" in ai_msgs[0].content


# ═══════════════════════════════════════════════════════════════════════════════
# should_proceed_to_enrichment
# ═══════════════════════════════════════════════════════════════════════════════

class TestShouldProceedToEnrichment:

    def test_returns_end_when_destination_missing(self):
        from travel_assistant.chatbot import should_proceed_to_enrichment
        state = make_state(itinerary_context={})
        assert should_proceed_to_enrichment(state) == "end"

    def test_returns_end_when_partially_filled(self):
        from travel_assistant.chatbot import should_proceed_to_enrichment
        ctx = {"destination": "Tokyo", "num_days": 5}  # missing 6 fields
        state = make_state(itinerary_context=ctx)
        assert should_proceed_to_enrichment(state) == "end"

    def test_returns_end_when_one_field_missing(self):
        from travel_assistant.chatbot import should_proceed_to_enrichment
        ctx = dict(FULL_CTX)
        del ctx["interests"]  # remove just one
        state = make_state(itinerary_context=ctx)
        assert should_proceed_to_enrichment(state) == "end"

    def test_returns_enrich_when_all_fields_present(self):
        from travel_assistant.chatbot import should_proceed_to_enrichment
        state = make_state(itinerary_context=dict(FULL_CTX))
        assert should_proceed_to_enrichment(state) == "enrich"

    def test_returns_end_when_context_is_none(self):
        from travel_assistant.chatbot import should_proceed_to_enrichment
        state = make_state(itinerary_context=None)
        assert should_proceed_to_enrichment(state) == "end"

    def test_returns_end_for_falsy_field_value(self):
        """A field set to None or empty string should count as missing."""
        from travel_assistant.chatbot import should_proceed_to_enrichment
        ctx = dict(FULL_CTX)
        ctx["cuisine"] = None
        state = make_state(itinerary_context=ctx)
        assert should_proceed_to_enrichment(state) == "end"


# ═══════════════════════════════════════════════════════════════════════════════
# enrich_itinerary_data
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnrichItineraryData:

    def test_returns_empty_when_no_destination(self):
        from travel_assistant.chatbot import enrich_itinerary_data
        state = make_state(itinerary_context={})
        result = enrich_itinerary_data(state, FAKE_CONFIG, store=FakeStore())
        assert result["itinerary_places"] == []

    def test_returns_empty_when_geocode_fails(self):
        from travel_assistant.chatbot import enrich_itinerary_data
        state = make_state(itinerary_context={"destination": "NonExistentCity123"})

        with patch("travel_assistant.chatbot._geocode_city", return_value=None):
            result = enrich_itinerary_data(state, FAKE_CONFIG, store=FakeStore())

        assert result["itinerary_places"] == []

    def test_fetches_places_when_geocode_succeeds(self):
        from travel_assistant.chatbot import enrich_itinerary_data

        fake_places = [
            {"name": "Eiffel Tower", "type": "attraction", "distance": 200},
            {"name": "Le Jules Verne", "type": "restaurant", "distance": 210},
        ]
        state = make_state(itinerary_context={"destination": "Paris"})

        with patch("travel_assistant.chatbot._geocode_city", return_value={"lat": 48.85, "lng": 2.35}), \
             patch("travel_assistant.chatbot._fetch_destination_places", return_value=fake_places):
            result = enrich_itinerary_data(state, FAKE_CONFIG, store=FakeStore())

        assert result["itinerary_places"] == fake_places

    def test_enriches_context_with_coords(self):
        """dest_lat and dest_lng should be added to itinerary_context."""
        from travel_assistant.chatbot import enrich_itinerary_data

        state = make_state(itinerary_context={"destination": "Paris"})

        with patch("travel_assistant.chatbot._geocode_city", return_value={"lat": 48.85, "lng": 2.35}), \
             patch("travel_assistant.chatbot._fetch_destination_places", return_value=[]):
            result = enrich_itinerary_data(state, FAKE_CONFIG, store=FakeStore())

        ctx = result["itinerary_context"]
        assert ctx["dest_lat"] == 48.85
        assert ctx["dest_lng"] == 2.35

    def test_calculates_travel_distance_when_location_available(self):
        """travel_distance_km should be set when user location is known."""
        from travel_assistant.chatbot import enrich_itinerary_data

        state = make_state(
            itinerary_context={"destination": "Paris"},
            location={"lat": 51.5, "lng": -0.1},  # London
        )

        with patch("travel_assistant.chatbot._geocode_city", return_value={"lat": 48.85, "lng": 2.35}), \
             patch("travel_assistant.chatbot._fetch_destination_places", return_value=[]):
            result = enrich_itinerary_data(state, FAKE_CONFIG, store=FakeStore())

        ctx = result["itinerary_context"]
        assert ctx["travel_distance_km"] is not None
        assert isinstance(ctx["travel_distance_km"], float)
        assert ctx["travel_distance_km"] > 0

    def test_no_travel_distance_when_no_location(self):
        """travel_distance_km should be absent when no user location given."""
        from travel_assistant.chatbot import enrich_itinerary_data

        state = make_state(itinerary_context={"destination": "Paris"}, location=None)

        with patch("travel_assistant.chatbot._geocode_city", return_value={"lat": 48.85, "lng": 2.35}), \
             patch("travel_assistant.chatbot._fetch_destination_places", return_value=[]):
            result = enrich_itinerary_data(state, FAKE_CONFIG, store=FakeStore())

        ctx = result["itinerary_context"]
        assert ctx.get("travel_distance_km") is None

    def test_preserves_existing_context_fields(self):
        """Existing itinerary_context fields must survive enrichment."""
        from travel_assistant.chatbot import enrich_itinerary_data

        state = make_state(itinerary_context=dict(FULL_CTX))

        with patch("travel_assistant.chatbot._geocode_city", return_value={"lat": 48.85, "lng": 2.35}), \
             patch("travel_assistant.chatbot._fetch_destination_places", return_value=[]):
            result = enrich_itinerary_data(state, FAKE_CONFIG, store=FakeStore())

        ctx = result["itinerary_context"]
        assert ctx["num_days"] == FULL_CTX["num_days"]
        assert ctx["party_size"] == FULL_CTX["party_size"]
        assert ctx["cuisine"] == FULL_CTX["cuisine"]

    def test_passes_through_itinerary_messages(self):
        """itinerary_messages should be returned unchanged."""
        from travel_assistant.chatbot import enrich_itinerary_data

        msgs = [HumanMessage(content="Paris"), AIMessage(content="Great choice!")]
        state = make_state(itinerary_context={"destination": "Paris"})
        state["itinerary_messages"] = msgs

        with patch("travel_assistant.chatbot._geocode_city", return_value={"lat": 48.85, "lng": 2.35}), \
             patch("travel_assistant.chatbot._fetch_destination_places", return_value=[]):
            result = enrich_itinerary_data(state, FAKE_CONFIG, store=FakeStore())

        assert result["itinerary_messages"] == msgs