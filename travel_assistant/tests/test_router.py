"""
Tests for router logic:
  - router_decision  (pure state-reading function, no mocks needed)
  - router_node      (calls classify_intent — mock that, everything else is pure)

router_decision coverage:
  - Each action type maps to the correct graph node string
  - ASK_CLARIFICATION with previous results routes back to last handler
  - ASK_CLARIFICATION with no previous results routes to "clarification"
  - FALLBACK routes to "General Chat/ Fallback"
  - target == None routes to "General Chat/ Fallback"
  - INTENT_FALLBACK_GENERAL_TRAVEL routes to "General Chat/ Fallback"
  - All intent targets route to their own string

router_node coverage:
  - Active itinerary context bypasses classify_intent entirely
  - Active itinerary context returns INTENT_C routing
  - No itinerary context calls classify_intent
  - Result contains classification dict with expected keys
  - Result contains routing dict with expected keys
  - previous_intent is updated to current intent
  - Switching away from INTENT_C clears itinerary_context and itinerary_messages
  - Switching away from non-itinerary intent does NOT clear anything
  - Same intent on consecutive turns does NOT clear anything
"""

import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage

from travel_assistant.intents import Intent, IntentClassificationResult, RoutingAction


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


class FakeStore:
    def get(self, ns, key): return None
    def put(self, ns, key, value): pass


def make_state(**overrides):
    base = {
        "messages": [HumanMessage(content="test")],
        "location": None,
        "nearby_context": None,
        "time_context": {"local_time": "12:00", "day_of_week": "Monday", "hour": 12, "is_weekend": False},
        "preferences": {"budget": None, "cuisine": None, "vibe": None, "pace": None},
        "last_results": None,
        "location_history_text": "No location history yet",
        "itinerary_context": None,
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


def make_routing_state(action, target=None, last_results=None):
    """Build a state dict with routing already set — for router_decision tests."""
    return make_state(
        routing={"action": action, "target_intent": target},
        last_results=last_results,
    )


def make_classification(primary: Intent, confidence: float = 0.9) -> IntentClassificationResult:
    """Build a valid IntentClassificationResult with one dominant intent."""
    remainder = round((1.0 - confidence) / (len(Intent) - 1), 6)
    scores = {intent: remainder for intent in Intent}
    scores[primary] = confidence
    return IntentClassificationResult(
        primary_intent=primary,
        confidence=confidence,
        intent_scores=scores,
        needs_clarification=False,
        safety_override=(primary == Intent.INTENT_G_URGENT_HEALTH),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# router_decision — pure function, no mocks needed
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouterDecision:

    def test_urgent_override_routes_to_health_emergency(self):
        from travel_assistant.chatbot import router_decision

        state = make_routing_state("URGENT_OVERRIDE", "INTENT_G_URGENT_HEALTH")
        assert router_decision(state, FAKE_CONFIG, store=FakeStore()) == "Health Emergency"

    def test_fallback_action_routes_to_general_chat(self):
        from travel_assistant.chatbot import router_decision

        state = make_routing_state("FALLBACK", "INTENT_FALLBACK_GENERAL_TRAVEL")
        assert router_decision(state, FAKE_CONFIG, store=FakeStore()) == "General Chat/ Fallback"

    def test_none_target_routes_to_general_chat(self):
        from travel_assistant.chatbot import router_decision

        state = make_routing_state("ROUTE_INTENT", target=None)
        assert router_decision(state, FAKE_CONFIG, store=FakeStore()) == "General Chat/ Fallback"

    def test_fallback_intent_target_routes_to_general_chat(self):
        from travel_assistant.chatbot import router_decision

        state = make_routing_state("ROUTE_INTENT", "INTENT_FALLBACK_GENERAL_TRAVEL")
        assert router_decision(state, FAKE_CONFIG, store=FakeStore()) == "General Chat/ Fallback"

    def test_clarification_with_no_previous_results_routes_to_clarification(self):
        from travel_assistant.chatbot import router_decision

        state = make_routing_state("ASK_CLARIFICATION", last_results=None)
        assert router_decision(state, FAKE_CONFIG, store=FakeStore()) == "clarification"

    def test_clarification_with_empty_previous_results_routes_to_clarification(self):
        from travel_assistant.chatbot import router_decision

        state = make_routing_state("ASK_CLARIFICATION", last_results=[])
        assert router_decision(state, FAKE_CONFIG, store=FakeStore()) == "clarification"

    def test_clarification_with_previous_results_routes_back_to_last_handler(self):
        """Follow-up ambiguous message should re-use the previous handler, not ask again."""
        from travel_assistant.chatbot import router_decision

        last_results = [{"handler": "INTENT_A_NEARBY_GENERIC", "response": "Here are some cafes."}]
        state = make_routing_state("ASK_CLARIFICATION", last_results=last_results)
        assert router_decision(state, FAKE_CONFIG, store=FakeStore()) == "INTENT_A_NEARBY_GENERIC"

    def test_clarification_uses_first_result_handler(self):
        """When multiple previous results exist, should use the first one's handler."""
        from travel_assistant.chatbot import router_decision

        last_results = [
            {"handler": "INTENT_B_NEARBY_BY_NEED", "response": "Try a park."},
            {"handler": "INTENT_A_NEARBY_GENERIC", "response": "Here are cafes."},
        ]
        state = make_routing_state("ASK_CLARIFICATION", last_results=last_results)
        assert router_decision(state, FAKE_CONFIG, store=FakeStore()) == "INTENT_B_NEARBY_BY_NEED"

    @pytest.mark.parametrize("intent", [
        Intent.INTENT_A_NEARBY_GENERIC,
        Intent.INTENT_B_NEARBY_BY_NEED,
        Intent.INTENT_C_ITINERARY,
        Intent.INTENT_D_FOOD_DIETARY,
        Intent.INTENT_E_FRIENDS_BASED,
        Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP,
    ])
    def test_route_intent_maps_to_correct_node(self, intent):
        """Every non-fallback intent should route to its own string value."""
        from travel_assistant.chatbot import router_decision

        state = make_routing_state("ROUTE_INTENT", intent.value)
        result = router_decision(state, FAKE_CONFIG, store=FakeStore())
        assert result == intent.value


# ═══════════════════════════════════════════════════════════════════════════════
# router_node — mock classify_intent, everything else is pure
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouterNode:

    # ── itinerary bypass ──────────────────────────────────────────────────────

    def test_active_itinerary_context_bypasses_classification(self):
        """Non-empty itinerary_context must skip classify_intent entirely."""
        from travel_assistant.chatbot import router_node

        state = make_state(itinerary_context={"destination": "Paris"})

        with patch("travel_assistant.chatbot.classify_intent") as mock_classify:
            router_node(state, FAKE_CONFIG, store=FakeStore())
            mock_classify.assert_not_called()

    def test_active_itinerary_context_routes_to_intent_c(self):
        from travel_assistant.chatbot import router_node

        state = make_state(itinerary_context={"destination": "Paris"})
        result = router_node(state, FAKE_CONFIG, store=FakeStore())

        assert result["classification"]["primary_intent"] == Intent.INTENT_C_ITINERARY.value
        assert result["routing"]["target_intent"] == Intent.INTENT_C_ITINERARY.value

    def test_empty_itinerary_context_calls_classification(self):
        """Empty dict is falsy — classification should proceed normally."""
        from travel_assistant.chatbot import router_node

        state = make_state(itinerary_context={})
        classification = make_classification(Intent.INTENT_A_NEARBY_GENERIC)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification) as mock_classify:
            router_node(state, FAKE_CONFIG, store=FakeStore())
            mock_classify.assert_called_once()

    def test_none_itinerary_context_calls_classification(self):
        from travel_assistant.chatbot import router_node

        state = make_state(itinerary_context=None)
        classification = make_classification(Intent.INTENT_A_NEARBY_GENERIC)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification) as mock_classify:
            router_node(state, FAKE_CONFIG, store=FakeStore())
            mock_classify.assert_called_once()

    # ── result structure ──────────────────────────────────────────────────────

    def test_result_contains_classification_keys(self):
        from travel_assistant.chatbot import router_node

        state = make_state()
        classification = make_classification(Intent.INTENT_A_NEARBY_GENERIC)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification):
            result = router_node(state, FAKE_CONFIG, store=FakeStore())

        assert "primary_intent" in result["classification"]
        assert "confidence" in result["classification"]
        assert "intent_scores" in result["classification"]
        assert "needs_clarification" in result["classification"]
        assert "safety_override" in result["classification"]

    def test_result_contains_routing_keys(self):
        from travel_assistant.chatbot import router_node

        state = make_state()
        classification = make_classification(Intent.INTENT_A_NEARBY_GENERIC)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification):
            result = router_node(state, FAKE_CONFIG, store=FakeStore())

        assert "action" in result["routing"]
        assert "target_intent" in result["routing"]

    def test_previous_intent_updated_to_current(self):
        """After routing, previous_intent should reflect the current turn's intent."""
        from travel_assistant.chatbot import router_node

        state = make_state(previous_intent=Intent.INTENT_B_NEARBY_BY_NEED)
        classification = make_classification(Intent.INTENT_A_NEARBY_GENERIC)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification):
            result = router_node(state, FAKE_CONFIG, store=FakeStore())

        assert result["previous_intent"] == Intent.INTENT_A_NEARBY_GENERIC

    def test_classification_primary_intent_matches_input(self):
        from travel_assistant.chatbot import router_node

        state = make_state()
        classification = make_classification(Intent.INTENT_D_FOOD_DIETARY)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification):
            result = router_node(state, FAKE_CONFIG, store=FakeStore())

        assert result["classification"]["primary_intent"] == Intent.INTENT_D_FOOD_DIETARY.value

    # ── intent-switching cleanup ──────────────────────────────────────────────

    def test_switching_from_itinerary_clears_context(self):
        """Switching away from INTENT_C must reset itinerary_context."""
        from travel_assistant.chatbot import router_node

        state = make_state(
            previous_intent=Intent.INTENT_C_ITINERARY,
            itinerary_context={},  # empty = no active flow, so classify_intent runs
        )
        classification = make_classification(Intent.INTENT_A_NEARBY_GENERIC)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification):
            result = router_node(state, FAKE_CONFIG, store=FakeStore())

        assert result["itinerary_context"] == {}
        assert result["itinerary_messages"] == []

    def test_switching_from_itinerary_clears_messages(self):
        from travel_assistant.chatbot import router_node

        state = make_state(
            previous_intent=Intent.INTENT_C_ITINERARY,
            itinerary_context={},
        )
        state["itinerary_messages"] = [HumanMessage(content="old itinerary message")]
        classification = make_classification(Intent.INTENT_A_NEARBY_GENERIC)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification):
            result = router_node(state, FAKE_CONFIG, store=FakeStore())

        assert result["itinerary_messages"] == []

    def test_switching_between_non_itinerary_intents_does_not_clear(self):
        """Switching A → B should never touch itinerary state."""
        from travel_assistant.chatbot import router_node

        state = make_state(
            previous_intent=Intent.INTENT_A_NEARBY_GENERIC,
            itinerary_context={},
        )
        classification = make_classification(Intent.INTENT_B_NEARBY_BY_NEED)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification):
            result = router_node(state, FAKE_CONFIG, store=FakeStore())

        # itinerary_context should NOT appear in result (not cleared, not set)
        assert "itinerary_context" not in result
        assert "itinerary_messages" not in result

    def test_same_intent_consecutive_turns_does_not_clear(self):
        """Staying on the same intent must never wipe itinerary state."""
        from travel_assistant.chatbot import router_node

        state = make_state(
            previous_intent=Intent.INTENT_A_NEARBY_GENERIC,
            itinerary_context={},
        )
        classification = make_classification(Intent.INTENT_A_NEARBY_GENERIC)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification):
            result = router_node(state, FAKE_CONFIG, store=FakeStore())

        assert "itinerary_context" not in result
        assert "itinerary_messages" not in result

    def test_no_previous_intent_does_not_clear(self):
        """First turn has no previous_intent — clearing logic must not run."""
        from travel_assistant.chatbot import router_node

        state = make_state(previous_intent=None, itinerary_context={})
        classification = make_classification(Intent.INTENT_A_NEARBY_GENERIC)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification):
            result = router_node(state, FAKE_CONFIG, store=FakeStore())

        assert "itinerary_context" not in result
        assert "itinerary_messages" not in result

    def test_switching_to_itinerary_does_not_clear(self):
        """Switching INTO INTENT_C (not away from it) must not clear anything."""
        from travel_assistant.chatbot import router_node

        state = make_state(
            previous_intent=Intent.INTENT_A_NEARBY_GENERIC,
            itinerary_context={},
        )
        classification = make_classification(Intent.INTENT_C_ITINERARY)

        with patch("travel_assistant.chatbot.classify_intent", return_value=classification):
            result = router_node(state, FAKE_CONFIG, store=FakeStore())

        assert "itinerary_context" not in result
        assert "itinerary_messages" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# route_intent — pure function, no mocks needed
# All 5 branches tested directly
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouteIntent:

    def _make_classification(
        self,
        primary: Intent,
        confidence: float,
        safety_override: bool = False,
        second_intent: Intent = None,
        second_score: float = None,
    ) -> IntentClassificationResult:
        """
        Build an IntentClassificationResult with precise control over
        top and second scores — needed to test gap-based branching.
        """
        # Fill all intents with a tiny base score
        scores = {intent: 0.001 for intent in Intent}
        scores[primary] = confidence

        # Set second intent score explicitly if provided
        if second_intent and second_score is not None:
            scores[second_intent] = second_score

        return IntentClassificationResult(
            primary_intent=primary,
            confidence=confidence,
            intent_scores=scores,
            needs_clarification=False,
            safety_override=safety_override,
        )

    # ── Branch 1: safety override ─────────────────────────────────────────────

    def test_safety_override_returns_urgent_override_action(self):
        """When safety_override=True, must return URGENT_OVERRIDE
        regardless of scores."""
        from travel_assistant.router import route_intent

        classification = self._make_classification(
            primary=Intent.INTENT_G_URGENT_HEALTH,
            confidence=1.0,
            safety_override=True,
        )
        result = route_intent(classification)

        assert result.action == RoutingAction.URGENT_OVERRIDE

    def test_safety_override_targets_urgent_health(self):
        """URGENT_OVERRIDE must always target INTENT_G_URGENT_HEALTH."""
        from travel_assistant.router import route_intent

        classification = self._make_classification(
            primary=Intent.INTENT_G_URGENT_HEALTH,
            confidence=1.0,
            safety_override=True,
        )
        result = route_intent(classification)

        assert result.target_intent == Intent.INTENT_G_URGENT_HEALTH

    def test_safety_override_ignores_scores(self):
        """Safety override should fire even if confidence is low."""
        from travel_assistant.router import route_intent

        classification = self._make_classification(
            primary=Intent.INTENT_G_URGENT_HEALTH,
            confidence=0.3,    # low confidence — but safety_override=True
            safety_override=True,
        )
        result = route_intent(classification)

        assert result.action == RoutingAction.URGENT_OVERRIDE

    # ── Branch 2: ambiguous intent ────────────────────────────────────────────

    def test_small_gap_returns_ask_clarification(self):
        """When top and second scores are within AMBIGUITY_DELTA,
        should return ASK_CLARIFICATION."""
        from travel_assistant.router import route_intent
        from travel_assistant.config import AMBIGUITY_DELTA

        top_score = 0.5
        # Gap clearly INSIDE the boundary — not at it
        second_score = top_score - (AMBIGUITY_DELTA / 2)

        classification = self._make_classification(
            primary=Intent.INTENT_A_NEARBY_GENERIC,
            confidence=top_score,
            second_intent=Intent.INTENT_B_NEARBY_BY_NEED,
            second_score=second_score,
        )
        result = route_intent(classification)

        assert result.action == RoutingAction.ASK_CLARIFICATION

    def test_ambiguous_result_has_no_target_intent(self):
        """ASK_CLARIFICATION result should have target_intent=None
        since we don't know which intent to route to."""
        from travel_assistant.router import route_intent
        from travel_assistant.config import AMBIGUITY_DELTA

        top_score = 0.5
        second_score = top_score - (AMBIGUITY_DELTA / 2)  # gap smaller than delta

        classification = self._make_classification(
            primary=Intent.INTENT_A_NEARBY_GENERIC,
            confidence=top_score,
            second_intent=Intent.INTENT_C_ITINERARY,
            second_score=second_score,
        )
        result = route_intent(classification)

        assert result.target_intent is None

    # ── Branch 3: high confidence ─────────────────────────────────────────────

    def test_high_confidence_returns_route_intent_action(self):
        """Score above HIGH_CONFIDENCE_THRESHOLD with clear gap
        should return ROUTE_INTENT."""
        from travel_assistant.router import route_intent
        from travel_assistant.config import HIGH_CONFIDENCE_THRESHOLD, AMBIGUITY_DELTA

        top_score = HIGH_CONFIDENCE_THRESHOLD + 0.05   # clearly above threshold
        second_score = top_score - (AMBIGUITY_DELTA + 0.1)  # clear gap

        classification = self._make_classification(
            primary=Intent.INTENT_D_FOOD_DIETARY,
            confidence=top_score,
            second_intent=Intent.INTENT_A_NEARBY_GENERIC,
            second_score=second_score,
        )
        result = route_intent(classification)

        assert result.action == RoutingAction.ROUTE_INTENT

    def test_high_confidence_targets_top_intent(self):
        """ROUTE_INTENT result should target whichever intent scored highest."""
        from travel_assistant.router import route_intent
        from travel_assistant.config import HIGH_CONFIDENCE_THRESHOLD, AMBIGUITY_DELTA

        top_score = HIGH_CONFIDENCE_THRESHOLD + 0.05
        second_score = top_score - (AMBIGUITY_DELTA + 0.1)

        classification = self._make_classification(
            primary=Intent.INTENT_D_FOOD_DIETARY,
            confidence=top_score,
            second_intent=Intent.INTENT_A_NEARBY_GENERIC,
            second_score=second_score,
        )
        result = route_intent(classification)

        assert result.target_intent == Intent.INTENT_D_FOOD_DIETARY

    @pytest.mark.parametrize("intent", [
        Intent.INTENT_A_NEARBY_GENERIC,
        Intent.INTENT_B_NEARBY_BY_NEED,
        Intent.INTENT_C_ITINERARY,
        Intent.INTENT_D_FOOD_DIETARY,
        Intent.INTENT_E_FRIENDS_BASED,
        Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP,
    ])
    def test_high_confidence_routes_each_intent(self, intent):
        """Every intent should be routable at high confidence."""
        from travel_assistant.router import route_intent
        from travel_assistant.config import HIGH_CONFIDENCE_THRESHOLD, AMBIGUITY_DELTA

        top_score = HIGH_CONFIDENCE_THRESHOLD + 0.05
        second_score = top_score - (AMBIGUITY_DELTA + 0.1)

        classification = self._make_classification(
            primary=intent,
            confidence=top_score,
            second_intent=Intent.INTENT_FALLBACK_GENERAL_TRAVEL,
            second_score=second_score,
        )
        result = route_intent(classification)

        assert result.action == RoutingAction.ROUTE_INTENT
        assert result.target_intent == intent

    # ── Branch 4: low confidence ──────────────────────────────────────────────

    def test_low_confidence_returns_fallback_action(self):
        """Score below LOW_CONFIDENCE_THRESHOLD should return FALLBACK."""
        from travel_assistant.router import route_intent
        from travel_assistant.config import LOW_CONFIDENCE_THRESHOLD, AMBIGUITY_DELTA

        top_score = LOW_CONFIDENCE_THRESHOLD - 0.05    # clearly below threshold
        second_score = top_score - (AMBIGUITY_DELTA + 0.1)  # clear gap

        classification = self._make_classification(
            primary=Intent.INTENT_A_NEARBY_GENERIC,
            confidence=top_score,
            second_intent=Intent.INTENT_FALLBACK_GENERAL_TRAVEL,
            second_score=second_score,
        )
        result = route_intent(classification)

        assert result.action == RoutingAction.FALLBACK

    def test_low_confidence_targets_fallback_intent(self):
        """FALLBACK action must target INTENT_FALLBACK_GENERAL_TRAVEL."""
        from travel_assistant.router import route_intent
        from travel_assistant.config import LOW_CONFIDENCE_THRESHOLD, AMBIGUITY_DELTA

        top_score = LOW_CONFIDENCE_THRESHOLD - 0.05
        second_score = top_score - (AMBIGUITY_DELTA + 0.1)

        classification = self._make_classification(
            primary=Intent.INTENT_A_NEARBY_GENERIC,
            confidence=top_score,
            second_intent=Intent.INTENT_FALLBACK_GENERAL_TRAVEL,
            second_score=second_score,
        )
        result = route_intent(classification)

        assert result.target_intent == Intent.INTENT_FALLBACK_GENERAL_TRAVEL

    # ── Branch 5: default (middle confidence) ─────────────────────────────────

    def test_middle_confidence_returns_route_intent(self):
        """Score between LOW and HIGH thresholds with clear gap
        should still route to top intent — the default branch."""
        from travel_assistant.router import route_intent
        from travel_assistant.config import (
            LOW_CONFIDENCE_THRESHOLD,
            HIGH_CONFIDENCE_THRESHOLD,
            AMBIGUITY_DELTA,
        )

        # Score sits in the middle band
        top_score = (LOW_CONFIDENCE_THRESHOLD + HIGH_CONFIDENCE_THRESHOLD) / 2
        second_score = top_score - (AMBIGUITY_DELTA + 0.1)

        classification = self._make_classification(
            primary=Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP,
            confidence=top_score,
            second_intent=Intent.INTENT_FALLBACK_GENERAL_TRAVEL,
            second_score=second_score,
        )
        result = route_intent(classification)

        assert result.action == RoutingAction.ROUTE_INTENT
        assert result.target_intent == Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP

    def test_middle_confidence_targets_top_intent(self):
        """Default branch routes to whichever intent had the highest score."""
        from travel_assistant.router import route_intent
        from travel_assistant.config import (
            LOW_CONFIDENCE_THRESHOLD,
            HIGH_CONFIDENCE_THRESHOLD,
            AMBIGUITY_DELTA,
        )

        top_score = (LOW_CONFIDENCE_THRESHOLD + HIGH_CONFIDENCE_THRESHOLD) / 2
        second_score = top_score - (AMBIGUITY_DELTA + 0.1)

        classification = self._make_classification(
            primary=Intent.INTENT_E_FRIENDS_BASED,
            confidence=top_score,
            second_intent=Intent.INTENT_FALLBACK_GENERAL_TRAVEL,
            second_score=second_score,
        )
        result = route_intent(classification)

        assert result.target_intent == Intent.INTENT_E_FRIENDS_BASED

    # ── Threshold boundary conditions ─────────────────────────────────────────

    def test_score_exactly_at_high_threshold_routes_directly(self):
        """Score exactly equal to HIGH_CONFIDENCE_THRESHOLD should
        trigger branch 3, not branch 4 or 5."""
        from travel_assistant.router import route_intent
        from travel_assistant.config import HIGH_CONFIDENCE_THRESHOLD, AMBIGUITY_DELTA

        top_score = HIGH_CONFIDENCE_THRESHOLD   # exactly at boundary
        second_score = top_score - (AMBIGUITY_DELTA + 0.1)

        classification = self._make_classification(
            primary=Intent.INTENT_A_NEARBY_GENERIC,
            confidence=top_score,
            second_intent=Intent.INTENT_FALLBACK_GENERAL_TRAVEL,
            second_score=second_score,
        )
        result = route_intent(classification)

        assert result.action == RoutingAction.ROUTE_INTENT

    def test_score_exactly_at_low_threshold_does_not_fallback(self):
        """Score exactly equal to LOW_CONFIDENCE_THRESHOLD should NOT
        trigger fallback — the condition is strictly less than."""
        from travel_assistant.router import route_intent
        from travel_assistant.config import LOW_CONFIDENCE_THRESHOLD, AMBIGUITY_DELTA

        top_score = LOW_CONFIDENCE_THRESHOLD    # exactly at boundary
        second_score = top_score - (AMBIGUITY_DELTA + 0.1)

        classification = self._make_classification(
            primary=Intent.INTENT_A_NEARBY_GENERIC,
            confidence=top_score,
            second_intent=Intent.INTENT_FALLBACK_GENERAL_TRAVEL,
            second_score=second_score,
        )
        result = route_intent(classification)

        # Exactly at LOW threshold → falls through to default branch 5
        assert result.action == RoutingAction.ROUTE_INTENT