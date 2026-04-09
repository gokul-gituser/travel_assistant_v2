
import pytest
from travel_assistant.chatbot import _geocode_city, _fetch_destination_places

def test_geocode_city_success(monkeypatch):
    class FakeResponse:
        def json(self):
            return [{"lat": "10.0", "lon": "20.0"}]

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("travel_assistant.chatbot.requests.get", fake_get)

    result = _geocode_city("Test City")

    assert result == {"lat": 10.0, "lng": 20.0}

def test_geocode_city_no_results(monkeypatch):
    class FakeResponse:
        def json(self):
            return []

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("travel_assistant.chatbot.requests.get", fake_get)

    result = _geocode_city("Unknown")

    assert result is None


def test_geocode_city_exception(monkeypatch):
    def fake_get(*args, **kwargs):
        raise Exception("API failure")

    monkeypatch.setattr("travel_assistant.chatbot.requests.get", fake_get)

    result = _geocode_city("Error City")

    assert result is None


def test_fetch_places_success(monkeypatch):
    fake_data = {
        "elements": [
            {
                "tags": {"name": "Place A", "tourism": "attraction"},
                "lat": 10.0,
                "lon": 10.0,
            },
            {
                "tags": {"name": "Place B", "amenity": "restaurant"},
                "lat": 10.1,
                "lon": 10.1,
            },
        ]
    }

    class FakeResponse:
        status_code = 200

        def json(self):
            return fake_data

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("travel_assistant.chatbot.requests.post", fake_post)

    result = _fetch_destination_places(10.0, 10.0)

    assert len(result) == 2
    assert result[0]["name"] == "Place A"
    assert "distance" in result[0]


def test_fetch_places_dedup(monkeypatch):
    fake_data = {
        "elements": [
            {
                "tags": {"name": "Same Place", "tourism": "attraction"},
                "lat": 10.0,
                "lon": 10.0,
            },
            {
                "tags": {"name": "Same Place", "tourism": "attraction"},
                "lat": 10.0,
                "lon": 10.0,
            },
        ]
    }

    class FakeResponse:
        status_code = 200

        def json(self):
            return fake_data

    monkeypatch.setattr("travel_assistant.chatbot.requests.post", fake_post := lambda *a, **k: FakeResponse())

    result = _fetch_destination_places(10.0, 10.0)

    assert len(result) == 1


def test_fetch_places_skip_invalid(monkeypatch):
    fake_data = {
        "elements": [
            {"tags": {}, "lat": 10.0, "lon": 10.0},  # no name
        ]
    }

    class FakeResponse:
        status_code = 200

        def json(self):
            return fake_data

    monkeypatch.setattr("travel_assistant.chatbot.requests.post", lambda *a, **k: FakeResponse())

    result = _fetch_destination_places(10.0, 10.0)

    assert result == []


def test_fetch_places_api_failure(monkeypatch):
    class FakeResponse:
        status_code = 500

    monkeypatch.setattr("travel_assistant.chatbot.requests.post", lambda *a, **k: FakeResponse())

    result = _fetch_destination_places(10.0, 10.0)

    assert result == []


