# tests/test_utils.py

from unittest.mock import patch
from datetime import datetime
from travel_assistant.chatbot import (
    _calculate_distance,
    _format_places_for_llm,
    build_time_context,
    get_user_profile_text,
    get_travel_history_text,
)


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


def test_calculate_distance_zero():
    d = _calculate_distance(10.0, 10.0, 10.0, 10.0)
    assert d == 0


def test_calculate_distance_positive():
    d = _calculate_distance(10.0, 10.0, 10.1, 10.1)
    assert d > 0


def test_format_places_basic(sample_places):
    result = _format_places_for_llm(sample_places)

    assert "REAL PLACES IN DESTINATION" in result
    assert "Place A" in result
    assert "Place B" in result


def test_format_places_grouping(sample_places):
    result = _format_places_for_llm(sample_places)

    # restaurant → should go under RESTAURANTS
    assert "RESTAURANTS" in result.upper()

    # park → should go under PARKS
    assert "PARKS" in result.upper()


def test_format_places_empty():
    result = _format_places_for_llm([])

    assert "REAL PLACES IN DESTINATION" in result



class TestBuildTimeContext:

    def test_returns_all_required_keys(self):
        """Result must contain all four keys that GraphState expects."""
        result = build_time_context()

        assert "local_time" in result
        assert "day_of_week" in result
        assert "hour" in result
        assert "is_weekend" in result

    def test_local_time_format(self):
        """local_time must be in HH:MM format."""
        result = build_time_context()

        # Must match HH:MM pattern
        parts = result["local_time"].split(":")
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()
        assert 0 <= int(parts[0]) <= 23
        assert 0 <= int(parts[1]) <= 59

    def test_hour_is_integer(self):
        """hour must be an integer between 0 and 23."""
        result = build_time_context()

        assert isinstance(result["hour"], int)
        assert 0 <= result["hour"] <= 23

    def test_is_weekend_is_bool(self):
        """is_weekend must be a boolean."""
        result = build_time_context()

        assert isinstance(result["is_weekend"], bool)

    def test_day_of_week_is_valid(self):
        """day_of_week must be a real day name."""
        result = build_time_context()

        valid_days = {
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        }
        assert result["day_of_week"] in valid_days

    def test_is_weekend_true_on_saturday(self):
        """weekday() == 5 is Saturday — is_weekend must be True."""
        # Saturday
        fake_saturday = datetime(2025, 1, 4, 14, 30)  # a real Saturday

        with patch("travel_assistant.chatbot.datetime") as mock_dt:
            mock_dt.now.return_value = fake_saturday
            result = build_time_context()

        assert result["is_weekend"] is True

    def test_is_weekend_true_on_sunday(self):
        """weekday() == 6 is Sunday — is_weekend must be True."""
        fake_sunday = datetime(2025, 1, 5, 10, 0)  # a real Sunday

        with patch("travel_assistant.chatbot.datetime") as mock_dt:
            mock_dt.now.return_value = fake_sunday
            result = build_time_context()

        assert result["is_weekend"] is True

    def test_is_weekend_false_on_monday(self):
        """weekday() == 0 is Monday — is_weekend must be False."""
        fake_monday = datetime(2025, 1, 6, 9, 0)  # a real Monday

        with patch("travel_assistant.chatbot.datetime") as mock_dt:
            mock_dt.now.return_value = fake_monday
            result = build_time_context()

        assert result["is_weekend"] is False

    def test_hour_matches_local_time(self):
        """The hour field must match the HH part of local_time."""
        fake_time = datetime(2025, 6, 10, 17, 45)  # 17:45

        with patch("travel_assistant.chatbot.datetime") as mock_dt:
            mock_dt.now.return_value = fake_time
            result = build_time_context()

        assert result["hour"] == 17
        assert result["local_time"] == "17:45"

    def test_day_of_week_matches_mocked_date(self):
        """day_of_week must reflect the mocked date, not system clock."""
        fake_wednesday = datetime(2025, 1, 8, 12, 0)  # a real Wednesday

        with patch("travel_assistant.chatbot.datetime") as mock_dt:
            mock_dt.now.return_value = fake_wednesday
            result = build_time_context()

        assert result["day_of_week"] == "Wednesday"


# ═══════════════════════════════════════════════════════════════════════════════
# get_user_profile_text
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetUserProfileText:

    def test_returns_no_profile_when_user_id_is_none(self):
        """No user_id → should return the no-profile fallback string
        without touching the store."""
        store = FakeStore()
        result = get_user_profile_text(store, None)

        assert result == "No user profile"

    def test_returns_no_profile_when_user_id_is_empty_string(self):
        """Empty string user_id is falsy — same as None."""
        store = FakeStore()
        result = get_user_profile_text(store, "")

        assert result == "No user profile"

    def test_returns_no_profile_when_store_has_no_entry(self):
        """Valid user_id but nothing in store → no profile string."""
        store = FakeStore()  # empty
        result = get_user_profile_text(store, "user-123")

        assert result == "No user profile"

    def test_returns_formatted_string_when_profile_exists(self):
        """When profile exists in store, should return a formatted string."""
        store = FakeStore()
        store.put(
            ("user_profile", "user-123"),
            "profile",
            {
                "user_name": "Alice",
                "age": "30",
                "location": "London",
                "interests": ["hiking", "museums"],
                "dislikes": ["crowds"],
                "additional_notes": "Prefers morning tours",
            }
        )

        result = get_user_profile_text(store, "user-123")

        assert "Alice" in result
        assert "London" in result
        assert "hiking" in result
        assert "museums" in result
        assert "crowds" in result
        assert "Prefers morning tours" in result

    def test_name_appears_in_output(self):
        """user_name must appear in the formatted text."""
        store = FakeStore()
        store.put(
            ("user_profile", "u1"),
            "profile",
            {
                "user_name": "Bob",
                "age": None,
                "location": None,
                "interests": [],
                "dislikes": [],
                "additional_notes": None,
            }
        )

        result = get_user_profile_text(store, "u1")

        assert "Bob" in result

    def test_handles_missing_optional_fields_gracefully(self):
        """Profile with only user_name set — all optional fields
        should fall back to their defaults without crashing."""
        store = FakeStore()
        store.put(
            ("user_profile", "u1"),
            "profile",
            {
                "user_name": "Charlie",
                # age, location, interests, dislikes, additional_notes all missing
            }
        )

        result = get_user_profile_text(store, "u1")

        # Should not crash and should still contain the name
        assert "Charlie" in result
        assert "Not provided" in result  # default for missing age/location

    def test_interests_joined_as_comma_separated(self):
        """Multiple interests should be joined with commas."""
        store = FakeStore()
        store.put(
            ("user_profile", "u1"),
            "profile",
            {
                "user_name": "Dana",
                "age": None,
                "location": None,
                "interests": ["art", "food", "hiking"],
                "dislikes": [],
                "additional_notes": None,
            }
        )

        result = get_user_profile_text(store, "u1")

        assert "art" in result
        assert "food" in result
        assert "hiking" in result

    def test_reads_from_correct_namespace(self):
        """Profile stored under a different user_id must NOT be returned."""
        store = FakeStore()
        store.put(
            ("user_profile", "other-user"),
            "profile",
            {"user_name": "Eve", "interests": [], "dislikes": [],
             "additional_notes": None, "age": None, "location": None}
        )

        # Request profile for a different user
        result = get_user_profile_text(store, "my-user")

        assert result == "No user profile"
        assert "Eve" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# get_travel_history_text
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetTravelHistoryText:

    def test_returns_no_history_when_user_id_is_none(self):
        """No user_id → no history fallback string."""
        store = FakeStore()
        result = get_travel_history_text(store, None)

        assert result == "No travel history"

    def test_returns_no_history_when_store_is_empty(self):
        """Valid user_id but no history in store → no history string."""
        store = FakeStore()
        result = get_travel_history_text(store, "user-123")

        assert result == "No travel history"

    def test_returns_formatted_string_for_single_trip(self):
        """One trip in history → formatted string with city, country, places."""
        store = FakeStore()
        store.put(
            ("travel_history", "user-123"),
            "history",
            [
                {
                    "number": 1,
                    "city": "Rome",
                    "country": "Italy",
                    "places_visited": ["Colosseum", "Vatican"],
                    "time_of_visit": "2024-06",
                    "hours_spent": 48,
                }
            ]
        )

        result = get_travel_history_text(store, "user-123")

        assert "Rome" in result
        assert "Italy" in result
        assert "Colosseum" in result
        assert "Vatican" in result
        assert "2024-06" in result
        assert "48" in result

    def test_multiple_trips_all_appear_in_output(self):
        """Multiple trips should all appear as separate lines."""
        store = FakeStore()
        store.put(
            ("travel_history", "user-123"),
            "history",
            [
                {
                    "number": 1,
                    "city": "Tokyo",
                    "country": "Japan",
                    "places_visited": ["Shibuya"],
                    "time_of_visit": "2024-03",
                    "hours_spent": 72,
                },
                {
                    "number": 2,
                    "city": "Paris",
                    "country": "France",
                    "places_visited": ["Louvre"],
                    "time_of_visit": "2024-07",
                    "hours_spent": 48,
                },
            ]
        )

        result = get_travel_history_text(store, "user-123")

        assert "Tokyo" in result
        assert "Paris" in result
        assert "Shibuya" in result
        assert "Louvre" in result

    def test_places_visited_joined_correctly(self):
        """Multiple places_visited should appear comma separated."""
        store = FakeStore()
        store.put(
            ("travel_history", "u1"),
            "history",
            [
                {
                    "number": 1,
                    "city": "London",
                    "country": "UK",
                    "places_visited": ["Big Ben", "Tower Bridge", "Hyde Park"],
                    "time_of_visit": "2023-12",
                    "hours_spent": 24,
                }
            ]
        )

        result = get_travel_history_text(store, "u1")

        assert "Big Ben" in result
        assert "Tower Bridge" in result
        assert "Hyde Park" in result

    def test_handles_empty_places_visited(self):
        """Trip with no places_visited should not crash."""
        store = FakeStore()
        store.put(
            ("travel_history", "u1"),
            "history",
            [
                {
                    "number": 1,
                    "city": "Berlin",
                    "country": "Germany",
                    "places_visited": [],   # empty
                    "time_of_visit": "2024-01",
                    "hours_spent": 10,
                }
            ]
        )

        result = get_travel_history_text(store, "u1")

        assert "Berlin" in result  # city still appears

    def test_reads_from_correct_namespace(self):
        """History stored under a different user_id must NOT be returned."""
        store = FakeStore()
        store.put(
            ("travel_history", "other-user"),
            "history",
            [
                {
                    "number": 1,
                    "city": "Madrid",
                    "country": "Spain",
                    "places_visited": [],
                    "time_of_visit": "2024-05",
                    "hours_spent": 20,
                }
            ]
        )

        result = get_travel_history_text(store, "my-user")

        assert result == "No travel history"
        assert "Madrid" not in result

    def test_output_is_multiline_for_multiple_trips(self):
        """Multiple trips should produce multiple lines in output."""
        store = FakeStore()
        store.put(
            ("travel_history", "u1"),
            "history",
            [
                {"number": 1, "city": "Oslo", "country": "Norway",
                 "places_visited": [], "time_of_visit": "2023-06", "hours_spent": 12},
                {"number": 2, "city": "Bergen", "country": "Norway",
                 "places_visited": [], "time_of_visit": "2023-07", "hours_spent": 8},
            ]
        )

        result = get_travel_history_text(store, "u1")
        lines = result.strip().split("\n")

        assert len(lines) == 2