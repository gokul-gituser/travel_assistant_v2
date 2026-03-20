import os
from pyexpat.errors import messages
from unittest import result
import uuid
import json
import logging
from typing import Annotated, Optional, Dict, List, TypedDict
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.graph import StateGraph, END, START, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore
from langgraph.store.base import BaseStore
from trustcall import create_extractor
from pydantic import BaseModel, Field, model_validator
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
#from prompts import INTENT_CLASSIFIER_PROMPT
from typing import Optional, List, Literal
from pydantic import BaseModel
from datetime import datetime

import math
import requests

from intents import Intent, IntentClassificationResult
from router import route_intent

load_dotenv()
logger = logging.getLogger(__name__)

# ── Overpass helpers ───────────────────────────────────────────────────────
 
def _calculate_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
 
 
def _geocode_city(city: str) -> Optional[Dict]:
    """City name → {lat, lng} via Nominatim."""
    try:
        res = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": "travel-assistant-app"},
            timeout=10,
        )
        results = res.json()
        if results:
            lat = float(results[0]["lat"])
            lng = float(results[0]["lon"])
            print(f"🌍 Geocoded '{city}' → ({lat}, {lng})")
            return {"lat": lat, "lng": lng}
    except Exception as e:
        print(f"⚠️ Geocode failed for '{city}': {e}")
    return None
 
 
def _fetch_destination_places(lat: float, lng: float) -> List[Dict]:
    """
    Fetch tourist-relevant places within 5km of destination centre.
    Covers attractions, museums, restaurants, cafes, parks.
    """
    query = f"""
    [out:json][timeout:30];
    (
      node(around:5000,{lat},{lng})["tourism"="attraction"];
      node(around:5000,{lat},{lng})["tourism"="museum"];
      node(around:5000,{lat},{lng})["amenity"="restaurant"];
      node(around:5000,{lat},{lng})["amenity"="cafe"];
      node(around:5000,{lat},{lng})["leisure"="park"];
      way(around:5000,{lat},{lng})["tourism"="attraction"];
      way(around:5000,{lat},{lng})["tourism"="museum"];
      way(around:5000,{lat},{lng})["leisure"="park"];
    );
    out center 60;
    """
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=35,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception as e:
        print(f"⚠️ Overpass itinerary fetch failed: {e}")
        return []
 
    places = []
    for el in data.get("elements", []):
        tags  = el.get("tags", {})
        name  = tags.get("name")
        if not name:
            continue
        plat = el.get("lat") or el.get("center", {}).get("lat")
        plng = el.get("lon") or el.get("center", {}).get("lon")
        if not plat or not plng:
            continue
        places.append({
            "name":          name,
            "type":          tags.get("tourism") or tags.get("amenity") or tags.get("leisure") or "place",
            "lat":           plat,
            "lng":           plng,
            "distance":      round(_calculate_distance(lat, lng, plat, plng), 0),
            "opening_hours": tags.get("opening_hours", ""),
            "cuisine":       tags.get("cuisine", ""),
        })
 
    # Sort by distance, deduplicate by name, cap at 40
    places.sort(key=lambda x: x["distance"])
    seen, unique = set(), []
    for p in places:
        if p["name"] not in seen:
            seen.add(p["name"])
            unique.append(p)

    print(f"\n🧪 OVERPASS RESULT for destination ({lat}, {lng})")
    print(f"Total places fetched: {len(places)}")

    for p in places[:10]:
        print(f" - {p['name']} ({p['type']}) | {p['distance']}m")
    return unique[:40]
 
 
def _format_places_for_llm(places: List[Dict]) -> str:
    """Group places by type and format as a compact block for the LLM."""
    groups: Dict[str, List] = {
        "attractions": [], "museums": [],
        "restaurants": [], "cafes":   [], "parks": [],
    }
    mapping = {
        "attraction": "attractions", "museum": "museums",
        "restaurant": "restaurants", "cafe":   "cafes", "park": "parks",
    }
    for p in places:
        bucket = mapping.get(p["type"], "attractions")
        groups[bucket].append(p)
 
    lines = ["REAL PLACES IN DESTINATION (OpenStreetMap data):"]
    for category, items in groups.items():
        if not items:
            continue
        lines.append(f"\n{category.upper()}:")
        for p in items[:10]:
            dist  = f"{int(p['distance'])}m" if p["distance"] < 1000 else f"{p['distance']/1000:.1f}km"
            extra = []
            if p.get("cuisine"):       extra.append(p["cuisine"])
            if p.get("opening_hours"): extra.append(p["opening_hours"])
            suffix = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"  - {p['name']}{suffix} — {dist} from city centre")
    return "\n".join(lines)
 


llm = ChatOpenAI(model="gpt-4o-mini")


class UserProfile(BaseModel):
    """Complete profile of a user"""
    user_name: str = Field(description="The user's preferred name")
    age: Optional[str] = Field(default=None, description="User's age")
    location: Optional[str] = Field(default=None, description="The user's home city or country. The place where user lives")
    interests: list = Field(default_factory=list, description="User's interests")
    dislikes: list = Field(default_factory=list, description="Things user dislikes")
    additional_notes: Optional[str] = Field(default=None, description="Other personal details")

# Create the extractor for memory
trustcall_extractor = create_extractor(
    llm,
    tools=[UserProfile],
    tool_choice="UserProfile",
)



TRUSTCALL_INSTRUCTION = """Extract information from the conversation and update the user profile.
IMPORTANT RULES:
1. Store ONLY what the user explicitly said - do NOT infer, assume, or add reasons
2. Keep information as close to the user's original words as possible
3. If the user provides full sentences, store those full sentences
4. If the user gives keywords, store those keywords
5. Do NOT make assumptions about WHY they like/dislike something unless they explicitly told you
6. Preserve the user's original phrasing and intent"""

_PLACES_LIST_RULES = """
PLACES LIST RULES — FOLLOW EXACTLY:
- Write ONE short friendly intro sentence (e.g. "Here are the nearest places to you in Seattle:").
- Do NOT list, name, or describe any individual place.
- After your intro sentence, output this token on its own line: <<<PLACES_LIST>>>
- The complete sorted place list will be inserted automatically after your response.
- Do not add anything after <<<PLACES_LIST>>>."""

# ── Param extraction ───────────────────────────────────────────────────────
 
_EXTRACTOR_PROMPT = """Extract itinerary planning details from this message.

User message: "{message}"

Return ONLY a raw JSON object — no markdown, no explanation:
{{
    "destination": null,
    "current_location": null,
    "num_days": null,
    "party_size": null,
    "transport_to": null,
    "transport_within": null,
    "cuisine": null,
    "interests": null
}}

Rules:
- destination: city or place name as a string, null if not mentioned
- current_location: where user is currently / traveling from, null if not mentioned
- num_days: integer number of days. Examples: "plan my day"=1, "5 day trip"=5, "weekend"=2, "week"=7, null if not mentioned
- party_size: integer (number of people). Examples: "solo"=1, "couple"=2, "with kids"=family size, null if not mentioned
- transport_to: how they'll get to the destination (fly, train, drive, bus, ship, etc.). null if not mentioned
- transport_within: how they'll get around IN the destination (walking, transit, metro, taxi, car, bike, etc.). null if not mentioned
- cuisine: food preferences (vegetarian, halal, vegan, gluten-free, local, Italian, Asian, etc.). null if not mentioned
- interests: what they enjoy doing (history, museums, nature, hiking, nightlife, beach, food, shopping, adventure, family activities, etc.). null if not mentioned

IMPORTANT:
- Only extract what is EXPLICITLY stated
- If a field is not mentioned, return null
- If the message is unclear about a field, return null — don't infer
- Return null, not empty string or false
"""


SYSTEM_PROMPT_NEARBY_GENERIC = """You are a helpful travel assistant that helps users find places near them such as cafes, clinics, hospitals, pharmacies, cinemas, parks, restaurants, attractions, and other venues.

You will receive contextual data about the user. This may include:
- The user's location
- A list of nearby places (if location was available)
- Previous conversation results
- User profile information

Follow these rules strictly.

--------------------------------

LOCATION RULES

If the user's location is NOT available in the context:
- Politely ask the user to share or enable their location so you can find nearby places.
- Do NOT recommend any places.
- Do NOT invent or guess places.

If the user's location IS available but no nearby places list is provided:
- Inform the user that you have their location.
- Ask what type of place they want to find (e.g., restaurants, cafes, hospitals, parks, etc.).

If BOTH location AND nearby places are provided:
{places_list_rules}

--------------------------------

USER PROFILE
{{user_profile}}

--------------------------------

LIVE LOCATION. Use this for place recommendations — ignore "Home" in user profile:
Location: {{location_context}}
NEARBY PLACES:
{{nearby_places}}

LOCATION HISTORY (user's previously visited locations, most recent last):
{{location_history}}

ADDITIONAL RULES

If the user asks for:
- "more options"
- "cheaper places"
- "something different"
- "another one"

Use the previous results to avoid repeating recommendations and refine the suggestions using the nearby places list.
PREVIOUS RESULTS
{{last_results}}

Never fabricate places or distances.
Only use the data provided in the context.""".format(places_list_rules=_PLACES_LIST_RULES)

SYSTEM_PROMPT_NEARBY_BY_NEED = """You are a helpful travel assistant specializing in finding nearby places.
The user might not explicitly ask for a type of place, but instead express a mood, situation, or need. 
Your job is to infer what kind of nearby place would best fit the user's expressed need and suggest accordingly. Do not infer user's needs beyond what they explicitly say, but do use the signals in their message to make the best possible suggestion. 

For example, if they say "I'm bored", you might suggest a nearby park or cafe or other places that might excite the user. 
If the user says something like "It's raining", you might suggest indoor places or activites. 
If the user says "I'm with kids", you might suggest family and/or kid friendly venues. 
If the user says "date night", you might suggest romantic restaurants or bars.

IMPORTANT: You will be provided with the user's exact current location (coordinates and city) and a list of real nearby places already fetched for you. 
- NEVER ask the user for their location — it is already provided below in the context.
- ALWAYS use the provided nearby places list
{places_list_rules}

User Profile:
{{user_profile}}

If the user says things that could include checking previous results like "show me cheaper ones", "more options", "something different" — 
use the previous results provided to refine your response accordingly.
Previous Results:
{{last_results}}

LIVE LOCATION. Use this for place recommendations — ignore "Home" in user profile:
Location: {{location_context}}
NEARBY PLACES:
{{nearby_places}}

LOCATION HISTORY (user's previously visited locations, most recent last):
{{location_history}}""".format(places_list_rules=_PLACES_LIST_RULES)

SYSTEM_PROMPT_FOOD = """You are a food and dining expert travel assistant.
You specialize in finding restaurants with dietary considerations.

User Profile:
{{user_profile}}

IMPORTANT: You will be provided with the user's exact current location (coordinates and city) and a list of real nearby places already fetched for you. 
- NEVER ask the user for their location — it is already provided below in the context.
- ALWAYS use the provided nearby places list
{places_list_rules}

If the user says things that could include checking previous results like "show me cheaper ones", "more options", "something different" — 
use the previous results provided to refine your response accordingly.
Previous Results:
{{last_results}}

LIVE LOCATION. Use this for place recommendations — ignore "Home" in user profile:
Location: {{location_context}}

NEARBY PLACES:
{{nearby_places}}""".format(places_list_rules=_PLACES_LIST_RULES)

 
SYSTEM_PROMPT_ITINERARY_V2 = """You are an expert travel planner creating personalised day-by-day itineraries.
 

You will receive real places fetched from OpenStreetMap for the destination.
Use ONLY these real places when naming specific venues. Never invent places.
 
OUTPUT FORMAT — follow this format:
  Day 1 — [short theme]
    09:00  Place Name (type) — one sentence why it fits
    11:00  Next Place — brief note
    13:00  Lunch: Restaurant Name — cuisine note
    ...
    19:00  Dinner: Restaurant Name — vibe note
 
  Day 2 — ...   ← repeat for ALL {num_days} days
  Day 3 — ...   ← repeat for ALL {num_days} days
 
  🗺️ Travel note: [distance from current location + how to get there]
 
RULES:
- CRITICAL: Generate exactly {num_days} days — no more, no less
- Use ONLY places from the list below
- One sentence max per place
- Group places by proximity to minimise walking
- Match restaurants to cuisine preference and budget
- If a category has no options, skip it gracefully
- If num_days exceeds available attractions, suggest half-day alternatives or nearby day trips
 
━━━ CONTEXT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User profile:
{user_profile}
 
Trip parameters:
  Destination:      {destination}
  Days:             {num_days}
  Party size:       {party_size}
  Budget:           {budget}
  City transport:   {transport_mode}
  From user's location: {travel_distance}
  Cuisine preference:   {cuisine}
 
Travel history (avoid re-recommending visited places):
{travel_history}
 
Previous results:
{last_results}
 
{places_block}
"""

SYSTEM_PROMPT_FRIENDS_BASED = """You are a travel assistant specializing in recommendations based on a user’s friends, social circle, or people they know.

Your role:
- Suggest places, activities, or restaurants that the user’s friends may have visited, liked, or recommended.
- If explicit friend data is available, use it directly.
- If friend data is NOT available or social media accounts are not connected:
  - Clearly state that you don’t currently have access to friends’ activity.
  - Gently suggest that connecting social or activity accounts could enable more personalized recommendations.
  - Do NOT pressure, persuade aggressively, or assume the user wants to connect accounts.

Guidelines:
- Be neutral and optional when mentioning account connections.
- Offer an alternative: ask the user to name a friend, city, or type of place their friends usually enjoy.
- Do not invent friends, preferences, or past activity.

User Profile:
{user_profile}

If the user says things that could include checking previous results like "show me cheaper ones", "more options", "something different" — 
use the previous results provided to refine your response accordingly.
Previous Results:
{last_results}


Travel History:
{travel_history}
"""


SYSTEM_PROMPT_SAFETY = """You are a trusted travel safety advisor.
You provide practical safety information and local travel tips.

User Profile:
{user_profile}

If the user says things that could include checking previous results like "show me cheaper ones", "more options", "something different" — 
use the previous results provided to refine your response accordingly.
Previous Results:
{last_results}

Travel History:
{travel_history}"""

SYSTEM_PROMPT_FALLBACK = """
You are a friendly, practical travel assistant. Your job is to help the user plan travel and make travel-related decisions.

Core behaviors:
- Be helpful, warm, and concise. Prefer actionable recommendations over long explanations.
- Always clarify ambiguity when the user’s intent or context is unclear.
- Ask 1–3 leading questions that narrow down intent and constraints, instead of asking many questions.

How to ask clarifying questions:
- First, briefly state the two most likely interpretations of the user’s request.
- Then ask targeted questions to choose the right path.
- Offer a default assumption if the user doesn’t answer (and clearly label it as an assumption).

Example (ambiguous request):
User: “I want breakfast for the next 3 days.”
You: “Quick check: do you want (1) a simple 3-day breakfast plan/itinerary, or (2) nearby breakfast places where you’re staying? Also, what city/area are you in, and do you have any dietary preferences?”



You will receive contextual data about the user. This may include:
- The user's location
- A list of nearby places (if location was available)
- Previous conversation results
- User profile information

Follow these rules strictly.

--------------------------------

LOCATION RULES

If the user's location is NOT available in the context:
- Politely ask the user to share or enable their location so you can find nearby places.
- Do NOT recommend any places.
- Do NOT invent or guess places.

If the user's location IS available but no nearby places list is provided:
- Inform the user that you have their location.
- Ask what type of place they want to find (e.g., restaurants, cafes, hospitals, parks, etc.).

If BOTH location AND nearby places are provided:
{places_list_rules}


User Profile:
{{user_profile}}

If the user says things that could include checking previous results like "show me cheaper ones", "more options", "something different" — 
use the previous results provided to refine your response accordingly.
Previous Results:
{{last_results}}

LIVE LOCATION. Use this for place recommendations — ignore "Home" in user profile:
Location: {{location_context}}

LOCATION HISTORY (user's previously visited locations, most recent last):
{{location_history}}

NEARBY PLACES
{{nearby_places}}""".format(places_list_rules=_PLACES_LIST_RULES)

INTENT_CLASSIFIER_PROMPT = """
You are an expert intent detection engine for a travel assistant chatbot.

Your task is to analyze user messages and classify them into one of these intents based on the most important requirement provided
by the user. Some messages may contain contents related to multiple intents. In that case, do not infer that one intent signal should
be given more priority, just give scores to multiple intents. DO NOT INFER



INTENT DEFINITIONS:

INTENT_A_NEARBY_GENERIC:
- User asks for specific places or categories near them
- Examples: cafes, parks, pharmacies, cinemas, restaurants
- Triggered by concrete nouns and "near me"

INTENT_B_NEARBY_BY_NEED:
- User expresses a situation, mood, or constraint
- Examples: bored, raining, with kids, date night, free things, something fun
- The category is inferred, not explicitly stated

INTENT_C_ITINERARY:
- User wants a structured plan for a day or multiple days
- Examples: plan my day, itinerary, 2-day trip, what to do in [city]

INTENT_D_FOOD_DIETARY:
- User asks about food
- Examples: vegetarian, halal, gluten-free, looking for food, restaurant recommendations

INTENT_E_FRIENDS_BASED:
- User asks for recommendations based on friends' activity or preferences

INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP:
- User asks about safety, scams, local customs, transport, etiquette, practical info

INTENT_G_URGENT_HEALTH:
- Medical emergencies or symptoms requiring immediate attention

INTENT_FALLBACK_GENERAL_TRAVEL:
- Greetings ("hello", "hi")
- Non-travel statements ("I'm sleeping", "tell me something")
- Only use this when message is clearly NOT about travel at all




CRITICAL RULES:

1. Return scores for ALL intents, even if some are very low
2. All scores must be between 0.0 and 1.0
3. All scores should sum to 1.0
4.In some cases, text could belong to multiple intents. In such cases be honest about ambiguity: if multiple intents are plausible, assign MEANINGFUL scores to each
5. NEVER distribute scores evenly (like 0.12 to each intent)
6. For vague travel-related input (like "Suggest something fun" or "What should I do?"), 
   DO NOT default to FALLBACK. Instead, distribute scores among other intents based on what's plausible
7. For non-travel input (like "I'm sleeping", "hello"), THEN use high FALLBACK score


SCORING GUIDELINES:
- 0.90-1.00: Very clear, unambiguous intent
- 0.70-0.89: Strong signal, but check if other intents are close
- 0.50-0.69: Moderate signal, likely has competitors
- 0.30-0.49: Weak signal, plausible but not primary
- 0.10-0.29: Very weak signal, possible
- 0.00-0.09: No signal


CRITICAL REQUIREMENT:
You MUST return a score for EVERY single one of these intents, even if the score is 0.0:
{', '.join([intent.value for intent in Intent])}

Detect the intent from this message: {input}

If you forget any intent, your response will be rejected and you'll be asked to try again.
Make sure all_scores contains ALL {len(Intent)} intents.

Return your response with:
- all_scores: A dictionary with intent names as keys and scores (0.0-1.0) as values
- reasoning: Explanation of your scoring

"""


CONTEXT_EXTRACTOR_PROMPT = """You are a context extraction assistant for a travel chatbot.

Extract context signals from the user's message. Only extract what is explicitly stated.

Return a JSON object with these fields (use null if not mentioned):
{{
    "party_type": null,        // solo, couple, kids, group
    "party_size": null,        // number
    "mobility_needs": null,    // any accessibility requirements
    "budget": null,            // budget, mid-range, luxury
    "cuisine": null,           // type of food if mentioned
    "vibe": null,              // casual, romantic, lively, quiet etc
    "pace": null,              // slow, moderate, fast
    "radius_m": null,          // distance constraint in meters
    "open_now": null,          // true if user wants places open now
    "transport_mode": null     // walking, transit, driving
}}

User message: {message}

Return ONLY the JSON object, no other text."""

def _extract_params(llm, message: str) -> Dict:
    """
    Extract all 8 itinerary parameters from a user message.
    
    Returns a dict with keys: destination, current_location, num_days, party_size,
    transport_to, transport_within, cuisine, interests
    
    Only includes keys with non-null values.
    """
    
    try:
        resp = llm.invoke([HumanMessage(content=_EXTRACTOR_PROMPT.format(message=message))])
        data = json.loads(resp.content)
        
        # Validate we got a dict
        if not isinstance(data, dict):
            logger.warning(f"Extractor returned non-dict: {type(data)}")
            return {}
        
        # Filter to only include non-null, non-empty values
        cleaned = {}
        for k, v in data.items():
            if v is not None and v != "" and v != "null":
                # Handle numeric strings
                if k in ["num_days", "party_size"]:
                    try:
                        cleaned[k] = int(v) if isinstance(v, str) else v
                    except ValueError:
                        logger.warning(f"Could not convert {k}={v} to int")
                else:
                    cleaned[k] = v
        
        logger.info(f"✅ Extracted parameters: {list(cleaned.keys())}")
        return cleaned
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in parameter extraction: {e}")
        return {}
    except Exception as e:
        logger.error(f"⚠️ Parameter extraction failed: {e}")
        return {}

class IntentScores(BaseModel):
    """Structured intent classification output."""
    all_scores: dict = Field(
        description="Scores for all intents - keys must match Intent enum values"
    )
    reasoning: str = Field(description="Explanation of scoring")
    
    @model_validator(mode="after")
    def validate_intent_keys(self):
        """Ensure all_scores has all intent keys."""
        required_keys = {intent.value for intent in Intent}
        provided_keys = set(self.all_scores.keys())
        
        if not required_keys.issubset(provided_keys):
            missing = required_keys - provided_keys
            raise ValueError(f"Missing intent scores: {missing}")
        
        return self


DUMMY_TRAVEL_HISTORY = [
    {
        "number": 1,
        "country": "Japan",
        "city": "Tokyo",
        "places_visited": ["Shibuya Crossing", "Senso-ji Temple", "Tsukiji Market"],
        "time_of_visit": "2024-03",
        "hours_spent": 72
    },
    {
        "number": 2,
        "country": "France",
        "city": "Paris",
        "places_visited": ["Eiffel Tower", "Louvre Museum", "Montmartre"],
        "time_of_visit": "2024-07",
        "hours_spent": 48
    },
    {
        "number": 3,
        "country": "India",
        "city": "Mumbai",
        "places_visited": ["Gateway of India", "Marine Drive", "Dharavi"],
        "time_of_visit": "2025-01",
        "hours_spent": 36
    }
]


classifier_agent = create_agent(
    model=llm,
    system_prompt=INTENT_CLASSIFIER_PROMPT,
    tools=[],
    response_format=ProviderStrategy(IntentScores),
)

class LocationContext(TypedDict):
    lat: float
    lng: float
    accuracy_m: Optional[float]
    captured_at: Optional[str]
    city: Optional[str]

class TimeContext(TypedDict):
    local_time: str
    day_of_week: str
    hour: int
    is_weekend: bool

class PartyContext(TypedDict):
    type: Optional[str]        # solo, couple, kids, group
    size: Optional[int]
    mobility_needs: Optional[str]

class PreferencesContext(TypedDict):
    budget: Optional[str]      # budget, mid-range, luxury
    cuisine: Optional[str]
    vibe: Optional[str]
    pace: Optional[str]        # slow, moderate, fast

class ConstraintsContext(TypedDict):
    radius_m: Optional[int]
    open_now: Optional[bool]
    transport_mode: Optional[str]  # walking, transit, driving

class GraphState(TypedDict):
    # Core conversation
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Context spine
    location: Optional[LocationContext]
    nearby_context: Optional[str]
    time_context: Optional[TimeContext]
    party: Optional[PartyContext]
    preferences: Optional[PreferencesContext]
    constraints: Optional[ConstraintsContext]
    connected_accounts: Optional[Dict] 
    safety_mode: str  # "normal" | "urgent_health"
    last_results: Optional[List[Dict]]
    location_history_text: Optional[str]

    previous_intent: Optional[str]

    itinerary_context: Optional[Dict]      # destination, num_days, defaults
    itinerary_places:  Optional[List[Dict]] # real places from Overpass
    itinerary_messages: Annotated[List[BaseMessage], add_messages]

    # Intent routing
    classification: Optional[Dict]
    routing: Optional[Dict]
    clarification_attempts: int



def build_time_context() -> TimeContext:
    """Build time context from current system time"""
    now = datetime.now()
    return {
        "local_time": now.strftime("%H:%M"),
        "day_of_week": now.strftime("%A"),
        "hour": now.hour,
        "is_weekend": now.weekday() >= 5
    }


def context_builder(state: GraphState, config: RunnableConfig, *, store: BaseStore) -> Dict:
    """Build context from config (location) and conversation (preferences/party/constraints)"""
    
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    user_msg = state["messages"][-1].content

    # . Location from config 
    raw_location = configurable.get("location")
    location = LocationContext(
        lat=raw_location["lat"],
        lng=raw_location["lng"],
        accuracy_m=raw_location.get("accuracy_m"),
        captured_at=raw_location.get("captured_at"),
        city=raw_location.get("city")
    ) if raw_location else None
    nearby_context = configurable.get("nearby_context")


    #  Time context from system clock
    time_context = build_time_context()

    connected_accounts = configurable.get("connected_accounts", {
        "google": False,
        "facebook": False,
        "instagram": False
    })

    last_results = state.get("last_results", None)
    print(f"DEBUG last_results: {last_results}")

    #. Extract party/preferences/constraints from user message via LLM
    extracted = {}
    try:
        response = llm.invoke([
            HumanMessage(content=CONTEXT_EXTRACTOR_PROMPT.format(message=user_msg))
        ])
        import json
        extracted = json.loads(response.content)
    except Exception as e:
        logger.warning(f"Context extraction failed: {e}")

    prev_party = state.get("party") or {}

    party = PartyContext(
        type=extracted.get("party_type") or prev_party.get("type"),
        size=extracted.get("party_size") or prev_party.get("size"),
        mobility_needs=extracted.get("mobility_needs") or prev_party.get("mobility_needs"),
    )

    prev_preferences = state.get("preferences") or {}
    preferences = PreferencesContext(
        budget=extracted.get("budget") or prev_preferences.get("budget"),
        cuisine=extracted.get("cuisine") or prev_preferences.get("cuisine"),
        vibe=extracted.get("vibe") or prev_preferences.get("vibe"),
        pace=extracted.get("pace") or prev_preferences.get("pace"),
    )

    prev_constraints = state.get("constraints") or {}
    constraints = ConstraintsContext(
        radius_m=extracted.get("radius_m") or prev_constraints.get("radius_m"),
        open_now=extracted.get("open_now") or prev_constraints.get("open_now"),
        transport_mode=extracted.get("transport_mode") or prev_constraints.get("transport_mode")
    )
    print(f"\n--- CONTEXT BUILT ---")
    print(f"Location: {location}")
    print(f"Time: {time_context}")
    print(f"Party: {party}")
    print(f"Preferences: {preferences}")
    print(f"Constraints: {constraints}")
    print(f"---------------------\n")

    history_namespace = ("location_history", user_id)
    existing_history = store.get(history_namespace, "history")
    location_history = existing_history.value if existing_history else []
    location_history_text = "\n".join([
        f"{h['date']} {h['time']} — {h['address']} ({h['lat']}, {h['lon']})"
        for h in location_history[-5:]  # last 5 
    ]) or "No location history yet"

    return {
        "location": location,
        "nearby_context": nearby_context,
        "time_context": time_context,
        "party": party,
        "preferences": preferences,
        "constraints": constraints,
        "connected_accounts": connected_accounts,
        "safety_mode": "normal",
        "last_results": last_results,
        "location_history_text": location_history_text,
    }


def classify_intent(user_input: str) -> IntentClassificationResult:
    """Classify user intent using agent with structured output."""
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            result = classifier_agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            
            scores_output = result.get("structured_response")
            
            if not scores_output:
                raise ValueError("No structured response from agent")
            
            # Check if all intents have scores
            required_keys = {intent.value for intent in Intent}
            provided_keys = set(scores_output.all_scores.keys())
            missing = required_keys - provided_keys
            
            if missing:
                retry_count += 1
                logger.warning(f"Missing scores for {missing}. Retry {retry_count}/{max_retries}")
                continue  # Retry
            
            # Parse the all_scores dict
            intent_scores = {}
            for intent_enum in Intent:
                intent_name = intent_enum.value
                score = scores_output.all_scores.get(intent_name, 0.0)
                intent_scores[intent_enum] = float(score)
            
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            confidence = intent_scores[primary_intent]
            safety_override = primary_intent == Intent.INTENT_G_URGENT_HEALTH
            
            logger.info(f"Classified as {primary_intent.value} (confidence: {confidence:.2f})")
            
            return IntentClassificationResult(
                primary_intent=primary_intent,
                confidence=confidence,
                intent_scores=intent_scores,
                needs_clarification=False,
                clarification_reason=None,
                safety_override=safety_override,
            )
        
        except Exception as e:
            retry_count += 1
            logger.warning(f"Classification attempt {retry_count} failed: {e}")
            
            if retry_count >= max_retries:
                logger.error(f"Classification failed after {max_retries} retries")
                break
    
    # Fallback after all retries exhausted
    logger.error("Intent Classification failed after all retries")
    intent_scores = {intent: 0.0 for intent in Intent}
    intent_scores[Intent.INTENT_FALLBACK_GENERAL_TRAVEL] = 1.0
    
    return IntentClassificationResult(
        primary_intent=Intent.INTENT_FALLBACK_GENERAL_TRAVEL,
        confidence=1.0,
        intent_scores=intent_scores,
        needs_clarification=False,
        clarification_reason=None,
        safety_override=False,
    )

def get_user_profile_text(store: BaseStore, user_id: str) -> str:
    """Retrieve and format cross-thread user profile"""
    if not user_id:
        return "No user profile"
    
    namespace = ("user_profile", user_id)
    existing = store.get(namespace, "profile")
    
    if not existing:
        return "No user profile"
    
    profile = existing.value
    return f"""Name: {profile.get('user_name', 'Unknown')}
            Age: {profile.get('age', 'Not provided')}
            Home: {profile.get('location', 'Not provided')}
            Interests: {', '.join(profile.get('interests', []))}
            Dislikes: {', '.join(profile.get('dislikes', []))}
            Notes: {profile.get('additional_notes', 'None')}"""


def get_travel_history_text(store: BaseStore, user_id: str) -> str:
    if not user_id:
        return "No travel history"
    
    namespace = ("travel_history", user_id)
    existing = store.get(namespace, "history")
    
    if not existing:
        return "No travel history"
    
    history = existing.value
    lines = []
    for trip in history:
        places = ", ".join(trip.get("places_visited", []))
        lines.append(
            f"{trip['number']}. {trip['city']}, {trip['country']} — "
            f"visited {places} in {trip['time_of_visit']} ({trip['hours_spent']} hrs)"
        )
    return "\n".join(lines)


def router_node(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    user_msg = state["messages"][-1].content

        # If already in itinerary flow, don't re-classify
    if state.get("itinerary_context"):
        return {
            "classification": {
                "primary_intent": Intent.INTENT_C_ITINERARY.value,
                "confidence": 1.0,
                "intent_scores": {},
                "needs_clarification": False,
                "safety_override": False,
            },
            "routing": {
                "action": "ROUTE",
                "target_intent": Intent.INTENT_C_ITINERARY.value,
            },
            "previous_intent": Intent.INTENT_C_ITINERARY,
        }

    classification = classify_intent(user_msg)
    decision = route_intent(classification)

    #state["classification"] = classification
    #state["routing"] = decision

    
    print(f"\n{'='*60}")
    print(f"Primary Intent: {classification.primary_intent.value}")
    print(f"Router Decision: {decision.action.name}")
    print(f"{'='*60}\n")

    current_intent = classification.primary_intent
    previous_intent = state.get("previous_intent")

    result = {
        "classification": {
            "primary_intent": classification.primary_intent.value,
            "confidence": classification.confidence,
            "intent_scores": {k.value: v for k, v in classification.intent_scores.items()},
            "needs_clarification": classification.needs_clarification,
            "safety_override": classification.safety_override,
        },
        "routing": {
            "action": decision.action.value,
            "target_intent": decision.target_intent.value if decision.target_intent else None,
        },
        "previous_intent": current_intent,  # Track for next turn
    }
    
    #Clear itinerary context when switching intents
    if previous_intent and previous_intent != current_intent:
        if previous_intent == Intent.INTENT_C_ITINERARY:
            logger.info(f"Switching from {previous_intent} to {current_intent}")
            logger.info(f"Clearing itinerary context and message history")
            
            # Reset itinerary-specific state
            result["itinerary_context"] = {}
            result["itinerary_messages"] = []
    
    return result

def router_decision(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    routing = state["routing"]
    action = routing["action"]
    target = routing.get("target_intent")

    if action == "URGENT_OVERRIDE":
        return "Health Emergency"

    if action == "ASK_CLARIFICATION":
        # If we have previous results, route back to same handler instead of clarifying
        last_results = state.get("last_results")
        if last_results:
            previous_handler = last_results[0].get("handler")
            if previous_handler:
                print(f"Follow-up detected — routing back to {previous_handler}")
                return previous_handler
        return "clarification"

    if action == "FALLBACK" or target is None or target == "INTENT_FALLBACK_GENERAL_TRAVEL":
        return "General Chat/ Fallback"
    
    return target


clarification_agent = create_agent(
    model=llm,
    system_prompt="""You are a clarification specialist for a travel chatbot.

The user's message is ambiguous and could mean multiple things.

Your job is to ask ONE clear, natural clarification question to help narrow down their actual intent.

Be friendly, concise, and ask only one targeted question.
Do NOT ask for general information - focus on disambiguating their intent.
Do NOT repeat questions they've already answered.""",
    tools=[],
)

def get_clarification_question(state: GraphState, config: RunnableConfig, *, store: BaseStore) -> str:
    """Generate clarification question"""
    classification = state.get("classification")
    user_msg = state["messages"][-1].content
    
    if not classification:
        return "Could you tell me more about what you're looking for?"
    
    # Get top competing intents
    scores = classification["intent_scores"]
    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_intents = [f"{intent} (score: {score:.2f})" for intent, score in sorted_intents[:3]]

    
    try:
        result = clarification_agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"""User said: "{user_msg}"

This could mean:
{chr(10).join(f"- {intent}" for intent in top_intents)}

Ask ONE natural clarification question to help disambiguate what they really want."""
            }]
        })
        #print(f"DEBUG clarification result keys: {result.keys()}")
        #print(f"DEBUG clarification result: {result}")
        
        messages = result.get("messages", [])
        question = messages[-1].content if messages else "Could you tell me more about what you're looking for?"
        return question
    
    except Exception as e:
        logger.warning(f"Clarification LLM failed: {e}")
        return "Could you give me more details about what you're looking for?"


def handle_clarification(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Ask clarification question and end."""
    attempts = state.get("clarification_attempts", 0)
    state["clarification_attempts"] = attempts + 1

    print(f"DEBUG: In handle_clarification")
    print(f"DEBUG: state.get('classification') = {state.get('classification')}")
    #print(f"DEBUG: classification is None? {state.get('classification') is None}")

    if attempts >= 2:
        # Max attempts reached, show message and end
        state["messages"].append(
            AIMessage(content="I'm having trouble understanding. Next, let me help with general travel assistance.")
        )
    else:
        # Generate and ask clarification question
        question = get_clarification_question(state,config,store=store)
        state["messages"].append(AIMessage(content=question))
    
    return state
   


# ── Node 1: collect_itinerary_context ─────────────────────────────────────
COLLECTION_QUESTIONS = {
    "destination": "Where are you planning to travel to? (e.g., Paris, Tokyo, New York)",
    "current_location": "Where are you traveling from? What's your current location?",
    "num_days": "How many days are you planning for this trip?",
    "party_size": "How many people are traveling? (including yourself)",
    "transport_to": "How will you get to {destination}? (e.g., fly, train, drive, bus)",
    "transport_within": "How do you prefer to get around within {destination}? (walk, metro, taxi, car, bike)",
    "cuisine": "Do you have any food preferences? (e.g., vegetarian, halal, local cuisine, no preference)",
    "interests": "What are your main interests? (e.g., history, nature, nightlife, family activities, adventure, food, shopping)",
}

COLLECTION_ORDER = [
    "destination",
    "current_location", 
    "num_days",
    "party_size",
    "transport_to",
    "transport_within",
    "cuisine",
    "interests",
]

def collect_itinerary_context(state, config, *, store):
    """
    Multi-turn collection of itinerary parameters.
    
    Process:
    1. Extract parameters from latest user message using LLM
    2. Merge with existing context (don't overwrite confirmed values)
    3. Auto-fill current_location from device location if available
    4. Find first missing required field
    5. Ask for it, or proceed to enrichment if all collected
    """
 # Get existing context (from previous turns)
    ctx = state.get("itinerary_context") or {}
    itinerary_messages = state.get("itinerary_messages") or []

    logger.info(f"📦 Loaded context: {list(ctx.keys())}")
    logger.info(f"📝 Message history: {len(itinerary_messages)} messages")

    user_msg = state["messages"][-1].content
    # 1. Extract parameters from this message
    extracted = _extract_params(llm, user_msg)
    
    # 2. Merge extracted params (only if not already confirmed in ctx)
    for key, value in extracted.items():
        if not ctx.get(key):  # Don't overwrite if already set
            ctx[key] = value
    
    # 3. Auto-fill current_location from device location if available
    if not ctx.get("current_location") and state.get("location"):
        loc = state["location"]
        ctx["current_location"] = loc.get("city", f"{loc.get('lat')}, {loc.get('lng')}")
        logger.info(f"Auto-filled current_location from device: {ctx['current_location']}")
    
    # 4. Find first missing required field
    missing_field = None
    for field_name in COLLECTION_ORDER:
        if not ctx.get(field_name):
            missing_field = field_name
            break
    
    # 5. If missing field found, ask for it
    if missing_field:
        question = COLLECTION_QUESTIONS[missing_field]
        
        # Format question with known destination if needed
        if "{destination}" in question and ctx.get("destination"):
            question = question.format(destination=ctx["destination"])
        
        logger.info(f"❓ Asking for missing field: {missing_field}")
        ai_message = AIMessage(content=question)

         # Store messages for context
        itinerary_messages.append(HumanMessage(content=user_msg))
        itinerary_messages.append(ai_message)

        # Keep rolling window (max 10 messages = 5 pairs)
        if len(itinerary_messages) > 10:
            itinerary_messages = itinerary_messages[-10:]
        
        logger.info(f"📝 Message history: {len(itinerary_messages)} messages")
        
        return {
            "itinerary_context": ctx,
            "itinerary_messages": itinerary_messages,  # ← Return updated history
            "messages": [ai_message],
        }
    
    # All params collected!
    logger.info(f"✅ All itinerary context collected!")

    itinerary_messages.append(HumanMessage(content=user_msg))
    
    return {
        **state,
        "itinerary_context": ctx,
        "itinerary_messages": itinerary_messages,  # ← Return history
        }
 
 
def itinerary_collect_decision(state) -> str:
    """
    Decide whether to keep asking (if fields are missing) or proceed to enrichment.
    
    Returns:
    - "ask": Stay in collection, ask next question
    - "enrich": All fields collected, proceed to enrich_itinerary_data
    """
    ctx = state.get("itinerary_context", {})
    
    # Check if any required field is still missing
    for field_name in COLLECTION_ORDER:
        if not ctx.get(field_name):
            return "ask"
    
    # All fields present
    return "enrich"
 
 
# ── Node 2: enrich_itinerary_data ──────────────────────────────────────────
 
def enrich_itinerary_data(state, config, *, store):
    """
    Geocode destination → fetch real places from Overpass.
    Calculates travel distance from user's current location if available.
    """
    ctx      = state.get("itinerary_context") or {}
    itinerary_messages = state.get("itinerary_messages") or []
    location = state.get("location")
    dest     = ctx.get("destination", "")
 
    print(f"\n🌍 Enriching itinerary data for: '{dest}'")
    logger.info(f"Full itinerary context: {ctx}")
 
    if not dest:
        logger.error(f"❌ CRITICAL: No destination in context! ctx={ctx}")
        print("⚠️ No destination — skipping enrichment")
        return {"itinerary_places": []}
 
    dest_coords = _geocode_city(dest)
    if not dest_coords:
        print(f"⚠️ Could not geocode '{dest}' — proceeding without place data")
        return {"itinerary_places": []}
 
    # Travel distance from user's current location
    travel_distance_km = None
    if location:
        travel_distance_km = round(
            _calculate_distance(
                location["lat"], location["lng"],
                dest_coords["lat"], dest_coords["lng"],
            ) / 1000, 1
        )
        print(f"📏 Travel distance to {dest}: {travel_distance_km} km")
 
    
    raw_places = _fetch_destination_places(dest_coords["lat"], dest_coords["lng"])
    print(f"✅ Fetched {len(raw_places)} places for {dest}")
 
    return {
        "itinerary_places": raw_places,
        "itinerary_context": {
            **ctx,
            "dest_lat":            dest_coords["lat"],
            "dest_lng":            dest_coords["lng"],
            "travel_distance_km":  travel_distance_km,
        },
        "itinerary_messages": itinerary_messages,
    }
 
def should_proceed_to_enrichment(state) -> str:
    """Check if all itinerary fields collected."""
    ctx = state.get("itinerary_context", {})
    
    required_fields = [
        "destination", "current_location", "num_days", "party_size",
        "transport_to", "transport_within", "cuisine", "interests"
    ]
    
    for field in required_fields:
        if not ctx.get(field):
            return "end"  # Still collecting
    
    return "enrich"  # All collected, proceed
 

def handle_nearby_generic(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Find nearby places"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    nearby = state.get("nearby_context") or ""
    time_context = state.get("time_context")
    preferences = state.get("preferences")
    print("DEBUG: NEARBY CONTEXT SENT TO LLM:")
    print(nearby)

    last_results = state.get("last_results") 

    location_context = (
        f"{location.get('city')} (lat: {location.get('lat')}, lng: {location.get('lng')})"
        if location else "NOT AVAILABLE"
    )

    location_history_text = state.get("location_history_text", "No location history yet")

    
#     context_text = f"""
#     Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
#     Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
#     User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
#     {f"Real nearby places:{chr(10)}{nearby}" if nearby else ""}
# """
    
    system_prompt = SYSTEM_PROMPT_NEARBY_GENERIC.format(user_profile=user_profile_text,
    location_context=location_context
    ,last_results=last_results or "No previous results",
    location_history=location_history_text
    ,nearby_places=nearby if nearby else "NOT AVAILABLE")
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    print("\n===== AI RESPONSE =====")
    print(response.content)
    print("=======================\n")
    
    return {"messages": [AIMessage(content=response.content)],
            "last_results": [{"handler": Intent.INTENT_A_NEARBY_GENERIC.value, "response": response.content}]} 


def handle_nearby_by_need(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Suggest based on mood/situation"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    nearby = state.get("nearby_context") or ""
    time_context = state.get("time_context")
    preferences = state.get("preferences")

    location_context = (
        f"{location.get('city')} (lat: {location.get('lat')}, lng: {location.get('lng')})"
        if location else "NOT AVAILABLE"
    )

    location_history_text = state.get("location_history_text", "No location history yet")


    last_results = state.get("last_results") 
    
#     context_text = f"""
#     Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
#     Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
#     User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
#     {f"Real nearby places:{chr(10)}{nearby}" if nearby else ""}
# """
    
    system_prompt = SYSTEM_PROMPT_NEARBY_BY_NEED.format(
        user_profile=user_profile_text,
        nearby_places=nearby if nearby else "NOT AVAILABLE",
        location_context=location_context,
        location_history=location_history_text,
        last_results=last_results or "No previous results")
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    print("\n===== AI RESPONSE =====")
    print(response.content)
    print("=======================\n")
    
    return {"messages": [AIMessage(content=response.content)],
             "last_results": [{"handler": Intent.INTENT_B_NEARBY_BY_NEED.value, "response": response.content}]} 



# ── Node 3: handle_itinerary (enhanced) ───────────────────────────────────
 
def handle_itinerary(state, config, *, store):
    """
    Generate the itinerary using real Overpass place data + full user context.
    Replaces the original handle_itinerary entirely.
    """
 
    user_id             = config["configurable"].get("user_id")
    user_profile_text   = get_user_profile_text(store, user_id)
    travel_history_text = get_travel_history_text(store, user_id)
 
    ctx          = state.get("itinerary_context") or {}
    itinerary_messages = state.get("itinerary_messages") or []  # ← Load history
    
    raw_places   = state.get("itinerary_places") or []
    print(f"\n🧪 DEBUG — itinerary raw_places count: {len(raw_places)}")

    for p in raw_places[:10]:
        print(f" - {p['name']} ({p['type']}) | {p['distance']}m")
    last_results = state.get("last_results")
 
    places_block = (
        _format_places_for_llm(raw_places)
        if raw_places
        else "No place data available — use your general knowledge of the destination."
    )
    print("\n🧪 PLACES BLOCK SENT TO LLM:\n")
    print(places_block[:1000])  # truncate if large
 
    travel_km  = ctx.get("travel_distance_km")
    travel_str = f"{travel_km} km from your current location" if travel_km else "Distance not available"
 
    system_prompt = SYSTEM_PROMPT_ITINERARY_V2.format(
        user_profile   = user_profile_text,
        destination    = ctx.get("destination", "Unknown destination"),
        num_days       = ctx.get("num_days", 1),
        party_size     = ctx.get("party_size", 1),
        budget         = ctx.get("budget", "mid-range"),
        transport_mode = ctx.get("transport_within", "walking"),
        travel_distance = travel_str,
        cuisine        = ctx.get("cuisine") or "no preference",
        travel_history = travel_history_text,
        last_results   = last_results or "No previous results",
        places_block   = places_block,
    )

    # Add conversation context to prompt
    conversation_context = ""
    if itinerary_messages:
        logger.info(f"Including {len(itinerary_messages)} messages in context")
        conversation_context = "\n\n" + "="*60 + "\nCONVERSATION CONTEXT\n" + "="*60 + "\n"
        conversation_context += "How the user built this itinerary:\n\n"
        
        # Include last 10 messages
        for msg in itinerary_messages[-10:]:
            if isinstance(msg, HumanMessage):
                conversation_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                conversation_context += f"Assistant: {msg.content}\n"
        
        conversation_context += "\n" + "="*60 + "\nUse this to personalize your response.\n"
 
    enhanced_system_prompt = system_prompt + conversation_context

    response = llm.invoke([
        SystemMessage(content=enhanced_system_prompt),
        *state["messages"][-3:],
    ])
 
    print("\n===== AI RESPONSE (ITINERARY) =====")
    print(response.content[:400], "...")
    print("===================================\n")
 
    return {
        "messages": [AIMessage(content=response.content)],
        "itinerary_messages": itinerary_messages,  # ← Keep history
        "last_results": [{
            "handler":  Intent.INTENT_C_ITINERARY.value,
            "response": response.content,
        }],
    }
 

def handle_food_dietary(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Food & dietary recommendations"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    nearby = state.get("nearby_context") or ""
    time_context = state.get("time_context")
    preferences = state.get("preferences")
    last_results = state.get("last_results") 
    
    location_context = (
        f"{location.get('city')} (lat: {location.get('lat')}, lng: {location.get('lng')})"
        if location else "NOT AVAILABLE"
    )
    
#     context_text = f"""
#     Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
#     Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
#     User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
#     {f"Real nearby places:{chr(10)}{nearby}" if nearby else ""}
# """
    
    system_prompt = SYSTEM_PROMPT_FOOD.format(
            user_profile=user_profile_text,
            last_results=last_results or "No previous results",
            location_context=location_context,           
            nearby_places=nearby if nearby else "NOT AVAILABLE", 
        )    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    print("\n===== AI RESPONSE =====")
    print(response.content)
    print("=======================\n")
    
    return {"messages": [AIMessage(content=response.content)],
             "last_results": [{"handler": Intent.INTENT_D_FOOD_DIETARY.value, "response": response.content}]} 


def handle_friends_based(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Friend-based recommendations"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    nearby = state.get("nearby_context") or ""
    time_context = state.get("time_context")
    preferences = state.get("preferences")
    last_results = state.get("last_results") 
    
    context_text = f"""
    Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
    Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
    User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
    {f"Real nearby places:{chr(10)}{nearby}" if nearby else ""}
"""
    
    system_prompt = SYSTEM_PROMPT_FRIENDS_BASED.format(user_profile=user_profile_text,travel_history=travel_history_text,last_results=last_results or "No previous results") + context_text
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    print("\n===== AI RESPONSE =====")
    print(response.content)
    print("=======================\n")
    
    return {"messages": [AIMessage(content=response.content)],
             "last_results": [{"handler": Intent.INTENT_E_FRIENDS_BASED.value, "response": response.content}]} 


def handle_safety_practical(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Safety & practical travel help"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    nearby = state.get("nearby_context") or ""
    time_context = state.get("time_context")
    preferences = state.get("preferences")
    last_results = state.get("last_results") 
    
    context_text = f"""
    Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
    Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
    User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
    {f"Real nearby places:{chr(10)}{nearby}" if nearby else ""}
"""
    
    system_prompt = SYSTEM_PROMPT_SAFETY.format(user_profile=user_profile_text,travel_history=travel_history_text,last_results=last_results or "No previous results") + context_text
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    print("\n===== AI RESPONSE =====")
    print(response.content)
    print("=======================\n")
    
    return {"messages": [AIMessage(content=response.content)],
             "last_results": [{"handler": Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP.value, "response": response.content}]} 


def handle_fallback(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """General chat fallback"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    nearby = state.get("nearby_context") or ""
    time_context = state.get("time_context")
    preferences = state.get("preferences")
    last_results = state.get("last_results") 

    location_context = (
        f"{location.get('city')} (lat: {location.get('lat')}, lng: {location.get('lng')})"
        if location else "NOT AVAILABLE"
    )

    location_history_text = state.get("location_history_text", "No location history yet")

    
#     context_text = f"""
#     Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
#     Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
#     User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
#     {f"Real nearby places:{chr(10)}{nearby}" if nearby else ""}
# """
    
    system_prompt = SYSTEM_PROMPT_FALLBACK.format(
            user_profile=user_profile_text,
            last_results=last_results or "No previous results",
            location_context=location_context,           
            location_history=location_history_text,
            nearby_places=nearby if nearby else "NOT AVAILABLE",  
        )    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    print("\n===== AI RESPONSE =====")
    print(response.content)
    print("=======================\n")
    
    return {"messages": [AIMessage(content=response.content)]} 


def handle_urgent(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Health emergency"""
    state["messages"].append(
        AIMessage(content="🚨 This sounds serious. Please seek medical help immediately.")
    )
    return state


def write_memory(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Extract and save user profile from conversation"""
    user_id = config["configurable"].get("user_id")
    
    if not user_id:
        return state
    
    # Retrieve existing profile
    namespace = ("user_profile", user_id)
    existing_memory = store.get(namespace, "profile")
    existing_profile = {"UserProfile": existing_memory.value} if existing_memory and existing_memory.value else None #added the and part
    
    # Extract updated profile from conversation
    result = trustcall_extractor.invoke({
        "messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"],
        "existing": existing_profile
    })

    #added this block
    if not result["responses"]:
        logger.warning("Trustcall returned no responses, skipping memory update")
        return state
    
    # Save updated profile
    updated_profile = result["responses"][0].model_dump()
    store.put(namespace, "profile", updated_profile)

    return state
    """
    #location
    location = state.get("location")
    if location:
        history_namespace = ("location_history", user_id)
        existing_history = store.get(history_namespace, "history")
        history = existing_history.value if existing_history else []

        history.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "lat": location.get("lat"),
            "lon": location.get("lng"),
            "address": location.get("city", "Unknown"),
            "type": "current_location"   # Hotel, Restaurant 
        })

        store.put(history_namespace, "history", history)
        #print(f"DEBUG location history saved: {history}")

    #save travel history  ---
    travel_namespace = ("travel_history", user_id)
    existing_travel = store.get(travel_namespace, "history")

    if not existing_travel:
        store.put(travel_namespace, "history", DUMMY_TRAVEL_HISTORY)
        logger.info(f"✓ Seeded travel history for {user_id}")
    
    logger.info(f"✓ Saved user profile for {user_id}")

    pending_namespace = ("pending_travel", user_id)
    pending = store.get(pending_namespace, "candidate")

    if pending and pending.value and not pending.value.get("confirmed"):

        user_text = state["messages"][-1].content.lower()

        if "yes" in user_text:

            travel_namespace = ("travel_history", user_id)
            existing = store.get(travel_namespace, "history")
            history = existing.value if existing else []

            new_trip = {
                "number": len(history) + 1,
                "country": pending.value.get("country"),
                "city": pending.value.get("city"),
                "places_visited": [],
                "time_of_visit": datetime.now().strftime("%Y-%m"),
                "hours_spent": 0
            }

            history.append(new_trip)
            store.put(travel_namespace, "history", history)

            pending.value["confirmed"] = True
            store.put(pending_namespace, "candidate", pending.value)

        elif "no" in user_text:
            store.put(pending_namespace, "candidate", None)
    """ 


def _build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("context_builder", context_builder)
    builder.add_node("router", router_node)
    builder.add_node(Intent.INTENT_A_NEARBY_GENERIC.value, handle_nearby_generic)
    builder.add_node(Intent.INTENT_B_NEARBY_BY_NEED.value, handle_nearby_by_need)

    builder.add_node("itinerary_collect", collect_itinerary_context)
    builder.add_node("itinerary_enrich",  enrich_itinerary_data)

    builder.add_node(Intent.INTENT_C_ITINERARY.value, handle_itinerary)
    builder.add_node(Intent.INTENT_D_FOOD_DIETARY.value, handle_food_dietary)
    builder.add_node(Intent.INTENT_E_FRIENDS_BASED.value, handle_friends_based)
    builder.add_node(Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP.value, handle_safety_practical)
    builder.add_node("clarification", handle_clarification)
    builder.add_node("General Chat/ Fallback", handle_fallback)
    builder.add_node("Health Emergency", handle_urgent)
    builder.add_node("write_memory", write_memory)


    builder.add_edge(START, "context_builder")
    builder.add_edge("context_builder", "router")

    builder.add_conditional_edges(
        "router",
        router_decision,
        {
            Intent.INTENT_A_NEARBY_GENERIC.value: Intent.INTENT_A_NEARBY_GENERIC.value,
            Intent.INTENT_B_NEARBY_BY_NEED.value: Intent.INTENT_B_NEARBY_BY_NEED.value,
            Intent.INTENT_C_ITINERARY.value: "itinerary_collect",
            Intent.INTENT_D_FOOD_DIETARY.value: Intent.INTENT_D_FOOD_DIETARY.value,
            Intent.INTENT_E_FRIENDS_BASED.value: Intent.INTENT_E_FRIENDS_BASED.value,
            Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP.value: Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP.value,
            "clarification": "clarification",
            "General Chat/ Fallback": "General Chat/ Fallback",
            "Health Emergency": "Health Emergency",
        },
    )

    builder.add_conditional_edges(
        "itinerary_collect",
        should_proceed_to_enrichment,  # ✅ Correct function
        {
            "end": END,                    # ✅ Routes to END when asking
            "enrich": "itinerary_enrich"   # ✅ Routes to enrichment when ready
        }
    )
    builder.add_edge("itinerary_enrich", Intent.INTENT_C_ITINERARY.value)
    builder.add_edge("clarification", END)

    for node in [
        Intent.INTENT_A_NEARBY_GENERIC.value,
        Intent.INTENT_B_NEARBY_BY_NEED.value,
        Intent.INTENT_C_ITINERARY.value,
        Intent.INTENT_D_FOOD_DIETARY.value,
        Intent.INTENT_E_FRIENDS_BASED.value,
        Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP.value,
        "General Chat/ Fallback",
        "Health Emergency",
    ]:
        builder.add_edge(node, "write_memory")

    builder.add_edge("write_memory", END)



    REDIS_URI = os.getenv("REDIS_URL")

    with RedisSaver.from_conn_string(REDIS_URI) as checkpointer:
        checkpointer.setup()
        with RedisStore.from_conn_string(REDIS_URI) as store:
            store.setup()
            graph = builder.compile(checkpointer=checkpointer, store=store)

        """    mermaid = graph.get_graph().draw_mermaid()
            with open("chatbot_graph_1.mmd", "w", encoding="utf-8") as f:
                f.write(mermaid)
            print("Wrote: chatbot_graph_1.mmd")
"""
            

    return graph
    

_graph = None


def _get_graph():
    """Get or initialize the compiled graph (singleton pattern)"""
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph

DUMMY_LOCATION = {
        "lat": 51.5074,
        "lng": -0.1278,
        "accuracy_m": 18,
        "captured_at": "2026-02-09T05:12:00Z",
        "city": "London"
    }


async def run_travel_assistant(
    user_id: str,
    text: str,
    location: Optional[dict] = None,
    thread_id: Optional[str] = None,
    nearby_context: Optional[str] = None,
) -> str:

    graph = _get_graph()

    if thread_id is None:
        thread_id = f"travel-{user_id}"
    config = {
        "configurable": {
            "user_id": user_id,
            "thread_id": thread_id,
            "location": location,
            "nearby_context": nearby_context,
            "connected_accounts": {"google": False, "facebook": False, "instagram": False},
        }
    }
    final_ai_message = None
    for chunk in graph.stream(
        {"messages": [HumanMessage(content=text)]},
        config,
        stream_mode="values",
    ):
        last_msg = chunk["messages"][-1]
        if isinstance(last_msg, AIMessage):
            final_ai_message = last_msg
    return final_ai_message.content if final_ai_message else "No response generated."


if __name__ == "__main__":        
    print("\n🤖 Travel Assistant started. Type 'exit' to quit.\n")
    user_id = input("Enter user ID: ").strip()
    thread_id = str(uuid.uuid4())

    
            
    config = {
        "configurable": {
            "user_id": user_id,
            "thread_id": thread_id,
            "location": DUMMY_LOCATION,
            "connected_accounts": {      
                "google": False,
                "facebook": False,
                "instagram": False
            }
        }
    }
    graph = _get_graph()
    while True:
        user_input = input("You: ").strip()
                
        if not user_input:
            continue
                
        if user_input.lower() in {"exit", "quit"}:
            print("🤖 Goodbye!")
            break
                
        final_ai_message = None
                
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values",
        ):
            last_msg = chunk["messages"][-1]
            if isinstance(last_msg, AIMessage):
                final_ai_message = last_msg
                
        if final_ai_message:
            print(f"Bot: {final_ai_message.content}\n")

            