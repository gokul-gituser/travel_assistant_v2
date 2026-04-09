# tests/conftest.py
import sys
import os
from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
from unittest.mock import MagicMock
from datetime import datetime


# ─────────────────────────────────────────────
# Fake LLM
# ─────────────────────────────────────────────
class FakeLLM:
    def __init__(self, response_content=None):
        self.response_content = response_content or "{}"

    def invoke(self, *args, **kwargs):
        class Resp:
            def __init__(self, content):
                self.content = content
        return Resp(self.response_content)


# ─────────────────────────────────────────────
# Fake Store (replaces RedisStore)
# ─────────────────────────────────────────────
class FakeStore:
    def __init__(self):
        self.data = {}

    def get(self, namespace, key):
        value = self.data.get((tuple(namespace), key))
        if value is None:
            return None

        class Obj:
            def __init__(self, val):
                self.value = val

        return Obj(value)

    def put(self, namespace, key, value):
        self.data[(tuple(namespace), key)] = value




# ─────────────────────────────────────────────
# Sample Places
# ─────────────────────────────────────────────
@pytest.fixture
def sample_places():
    return [
        {
            "name": "Place A",
            "type": "restaurant",
            "lat": 10.0,
            "lng": 10.0,
            "distance": 100,
            "opening_hours": "9-5",
            "cuisine": "indian",
        },
        {
            "name": "Place B",
            "type": "park",
            "lat": 10.1,
            "lng": 10.1,
            "distance": 200,
            "opening_hours": "",
            "cuisine": "",
        },
    ]


# ─────────────────────────────────────────────
# Sample GraphState
# ─────────────────────────────────────────────
@pytest.fixture
def base_state():
    return {
        "messages": [],
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


# ─────────────────────────────────────────────
# Sample Config
# ─────────────────────────────────────────────
@pytest.fixture
def base_config():
    return {
        "configurable": {
            "user_id": "test-user",
            "thread_id": "test-thread",
            "location": None,
            "nearby_context": None,
        }
    }


# ─────────────────────────────────────────────
# Fake classifier_agent
# ─────────────────────────────────────────────
@pytest.fixture
def mock_classifier(monkeypatch):
    from travel_assistant.chatbot import classifier_agent

    def fake_invoke(*args, **kwargs):
        class FakeResponse:
            def __init__(self):
                self.all_scores = {
                    "INTENT_A_NEARBY_GENERIC": 0.9,
                    "INTENT_B_NEARBY_BY_NEED": 0.1,
                    "INTENT_C_ITINERARY": 0.0,
                    "INTENT_D_FOOD_DIETARY": 0.0,
                    "INTENT_E_FRIENDS_BASED": 0.0,
                    "INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP": 0.0,
                    "INTENT_G_URGENT_HEALTH": 0.0,
                    "INTENT_FALLBACK_GENERAL_TRAVEL": 0.0,
                }

        return {"structured_response": FakeResponse()}

    monkeypatch.setattr(classifier_agent, "invoke", fake_invoke)


# ─────────────────────────────────────────────
# Fake LLM patch (global)
# ─────────────────────────────────────────────
@pytest.fixture
def mock_llm(monkeypatch):
    
    fake = FakeLLM(response_content="{}")
    monkeypatch.setattr(ChatOpenAI, "invoke", fake.invoke)
    return fake


@pytest.fixture
def fake_store():
    return FakeStore()