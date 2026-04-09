# tests/test_intent.py

from travel_assistant.chatbot import _extract_params, classify_intent
from travel_assistant.intents import Intent

def test_extract_params_basic(monkeypatch):
    fake_response = """
    {
        "destination": "Paris",
        "current_location": null,
        "num_days": 3,
        "party_size": 2,
        "transport_to": "flight",
        "transport_within": "metro",
        "cuisine": "vegetarian",
        "interests": "museums"
    }
    """

    class FakeLLM:
        def invoke(self, *args, **kwargs):
            class R:
                content = fake_response
            return R()

    result = _extract_params(FakeLLM(), "Plan a 3 day trip to Paris")

    assert result["destination"] == "Paris"
    assert result["num_days"] == 3
    assert result["party_size"] == 2


def test_extract_params_filters_null(monkeypatch):
    fake_response = """
    {
        "destination": null,
        "current_location": null,
        "num_days": null,
        "party_size": null,
        "transport_to": null,
        "transport_within": null,
        "cuisine": null,
        "interests": null
    }
    """

    class FakeLLM:
        def invoke(self, *args, **kwargs):
            class R:
                content = fake_response
            return R()

    result = _extract_params(FakeLLM(), "random text")

    assert result == {}


def test_extract_params_invalid_json(monkeypatch):
    class FakeLLM:
        def invoke(self, *args, **kwargs):
            class R:
                content = "INVALID JSON"
            return R()

    result = _extract_params(FakeLLM(), "test")

    assert result == {}

def test_extract_params_numeric_string(monkeypatch):
    fake_response = """
    {
        "destination": "Tokyo",
        "current_location": null,
        "num_days": "5",
        "party_size": "2",
        "transport_to": null,
        "transport_within": null,
        "cuisine": null,
        "interests": null
    }
    """

    class FakeLLM:
        def invoke(self, *args, **kwargs):
            class R:
                content = fake_response
            return R()

    result = _extract_params(FakeLLM(), "trip")

    assert result["num_days"] == 5
    assert result["party_size"] == 2


def test_classify_intent_basic(monkeypatch):

    def fake_invoke(*args, **kwargs):
        class FakeStructured:
            all_scores = {
                "INTENT_A_NEARBY_GENERIC": 0.8,
                "INTENT_B_NEARBY_BY_NEED": 0.1,
                "INTENT_C_ITINERARY": 0.05,
                "INTENT_D_FOOD_DIETARY": 0.05,
                "INTENT_E_FRIENDS_BASED": 0.0,
                "INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP": 0.0,
                "INTENT_G_URGENT_HEALTH": 0.0,
                "INTENT_FALLBACK_GENERAL_TRAVEL": 0.0,
            }

        return {"structured_response": FakeStructured()}

    monkeypatch.setattr("travel_assistant.chatbot.classifier_agent.invoke", fake_invoke)

    result = classify_intent("Find cafes near me")

    assert result.primary_intent == Intent.INTENT_A_NEARBY_GENERIC
    assert result.confidence == 0.8

def test_classify_intent_missing_keys(monkeypatch):

    def bad_invoke(*args, **kwargs):
        class Bad:
            all_scores = {"INTENT_A_NEARBY_GENERIC": 1.0}  # incomplete
        return {"structured_response": Bad()}

    monkeypatch.setattr("travel_assistant.chatbot.classifier_agent.invoke", bad_invoke)

    result = classify_intent("test")

    assert result.primary_intent == Intent.INTENT_FALLBACK_GENERAL_TRAVEL
    assert result.confidence == 1.0


def test_classify_intent_exception(monkeypatch):

    def raise_error(*args, **kwargs):
        raise Exception("LLM failed")

    monkeypatch.setattr("travel_assistant.chatbot.classifier_agent.invoke", raise_error)

    result = classify_intent("test")

    assert result.primary_intent == Intent.INTENT_FALLBACK_GENERAL_TRAVEL


def test_classify_intent_urgent(monkeypatch):

    def fake_invoke(*args, **kwargs):
        class FakeStructured:
            all_scores = {
                "INTENT_A_NEARBY_GENERIC": 0.0,
                "INTENT_B_NEARBY_BY_NEED": 0.0,
                "INTENT_C_ITINERARY": 0.0,
                "INTENT_D_FOOD_DIETARY": 0.0,
                "INTENT_E_FRIENDS_BASED": 0.0,
                "INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP": 0.0,
                "INTENT_G_URGENT_HEALTH": 1.0,
                "INTENT_FALLBACK_GENERAL_TRAVEL": 0.0,
            }

        return {"structured_response": FakeStructured()}

    monkeypatch.setattr("travel_assistant.chatbot.classifier_agent.invoke", fake_invoke)

    result = classify_intent("I am bleeding heavily")

    assert result.safety_override is True