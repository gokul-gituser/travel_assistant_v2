import os
from pyexpat.errors import messages
from unittest import result
import uuid
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

from intents import Intent, IntentClassificationResult
from router import route_intent

load_dotenv()
logger = logging.getLogger(__name__)


llm = ChatOpenAI(model="gpt-4o-mini")


class UserProfile(BaseModel):
    """Complete profile of a user"""
    user_name: str = Field(description="The user's preferred name")
    age: Optional[str] = Field(default=None, description="User's age")
    location: Optional[str] = Field(default=None, description="User's city/country")
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



SYSTEM_PROMPT_NEARBY_GENERIC = """You are a helpful travel assistant specializing in finding places are thar are near to the user's location.
You help users discover cafes, clinics,hospitals, pharmacies,cinemas, parks, restaurants, attractions,  and other venues near their location.


User Profile:
{user_profile}

If the user says things that could include checking previous results like "show me cheaper ones", "more options", "something different" — 
use the previous results provided to refine your response accordingly.
Previous Results:
{last_results}

Travel History:
{travel_history}"""

SYSTEM_PROMPT_NEARBY_BY_NEED = """You are a helpful travel assistant specializing in finding nearby places.
The user might not explicitly ask for a type of place, but instead express a mood, situation, or need. 
Your job is to infer what kind of nearby place would best fit the user's expressed need and suggest accordingly. Do not infer user's needs beyond what they explicitly say, but do use the signals in their message to make the best possible suggestion. 

For example, if they say "I'm bored", you might suggest a nearby park or cafe or other places that might excite the user. 
If the user says something like "It's raining", you might suggest indoor places or activites. 
If the user says "I'm with kids", you might suggest family and/or kid friendly venues. 
If the user says "date night", you might suggest romantic restaurants or bars.

User Profile:
{user_profile}

If the user says things that could include checking previous results like "show me cheaper ones", "more options", "something different" — 
use the previous results provided to refine your response accordingly.
Previous Results:
{last_results}

Travel History:
{travel_history}"""

SYSTEM_PROMPT_FOOD = """You are a food and dining expert travel assistant.
You specialize in finding restaurants with dietary considerations.

User Profile:
{user_profile}

If the user says things that could include checking previous results like "show me cheaper ones", "more options", "something different" — 
use the previous results provided to refine your response accordingly.
Previous Results:
{last_results}


Travel History:
{travel_history}"""

SYSTEM_PROMPT_ITINERARY = """You are a travel planning expert.
You create detailed, personalized day plans and itineraries.

User Profile:
{user_profile}

If the user says things that could include checking previous results like "show me cheaper ones", "more options", "something different" — 
use the previous results provided to refine your response accordingly.
Previous Results:
{last_results}

Travel History:
{travel_history}"""

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

User Profile:
{user_profile}

If the user says things that could include checking previous results like "show me cheaper ones", "more options", "something different" — 
use the previous results provided to refine your response accordingly.
Previous Results:
{last_results}

Travel History:
{travel_history}"""

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
    time_context: Optional[TimeContext]
    party: Optional[PartyContext]
    preferences: Optional[PreferencesContext]
    constraints: Optional[ConstraintsContext]
    connected_accounts: Optional[Dict] 
    safety_mode: str  # "normal" | "urgent_health"
    last_results: Optional[List[Dict]]
    
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

    party = PartyContext(
        type=extracted.get("party_type"),
        size=extracted.get("party_size"),
        mobility_needs=extracted.get("mobility_needs")
    )

    preferences = PreferencesContext(
        budget=extracted.get("budget"),
        cuisine=extracted.get("cuisine"),
        vibe=extracted.get("vibe"),
        pace=extracted.get("pace")
    )

    constraints = ConstraintsContext(
        radius_m=extracted.get("radius_m"),
        open_now=extracted.get("open_now"),
        transport_mode=extracted.get("transport_mode")
    )
    print(f"\n--- CONTEXT BUILT ---")
    print(f"Location: {location}")
    print(f"Time: {time_context}")
    print(f"Party: {party}")
    print(f"Preferences: {preferences}")
    print(f"Constraints: {constraints}")
    print(f"---------------------\n")

    return {
        "location": location,
        "time_context": time_context,
        "party": party,
        "preferences": preferences,
        "constraints": constraints,
        "connected_accounts": connected_accounts,
        "safety_mode": "normal",
        "last_results": last_results,
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
            Location: {profile.get('location', 'Not provided')}
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

    classification = classify_intent(user_msg)
    decision = route_intent(classification)

    #state["classification"] = classification
    #state["routing"] = decision
    
    print(f"\n{'='*60}")
    print(f"Primary Intent: {classification.primary_intent.value}")
    print(f"Router Decision: {decision.action.name}")
    print(f"{'='*60}\n")

    return {
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
        }
    }

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
   

def handle_nearby_generic(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Find nearby places"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    time_context = state.get("time_context")
    preferences = state.get("preferences")

    last_results = state.get("last_results") 
    
    context_text = f"""
    Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
    Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
    User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
"""
    
    system_prompt = SYSTEM_PROMPT_NEARBY_GENERIC.format(user_profile=user_profile_text,travel_history=travel_history_text,last_results=last_results or "No previous results")  + context_text
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    
    return {"messages": [AIMessage(content=response.content)],
            "last_results": [{"handler": Intent.INTENT_A_NEARBY_GENERIC.value, "response": response.content}]} 


def handle_nearby_by_need(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Suggest based on mood/situation"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    time_context = state.get("time_context")
    preferences = state.get("preferences")

    last_results = state.get("last_results") 
    
    context_text = f"""
    Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
    Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
    User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
"""
    
    system_prompt = SYSTEM_PROMPT_NEARBY_BY_NEED.format(user_profile=user_profile_text,travel_history=travel_history_text,last_results=last_results or "No previous results") + context_text
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    
    return {"messages": [AIMessage(content=response.content)],
             "last_results": [{"handler": Intent.INTENT_B_NEARBY_BY_NEED.value, "response": response.content}]} 


def handle_itinerary(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Create day plans/itineraries"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    time_context = state.get("time_context")
    preferences = state.get("preferences")
    last_results = state.get("last_results") 
    
    context_text = f"""
    Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
    Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
    User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
"""
    
    system_prompt = SYSTEM_PROMPT_ITINERARY.format(user_profile=user_profile_text,travel_history=travel_history_text,last_results=last_results or "No previous results") + context_text
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    
    return {"messages": [AIMessage(content=response.content)],
             "last_results": [{"handler": Intent.INTENT_C_ITINERARY.value, "response": response.content}]} 


def handle_food_dietary(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Food & dietary recommendations"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    time_context = state.get("time_context")
    preferences = state.get("preferences")
    last_results = state.get("last_results") 
    
    context_text = f"""
    Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
    Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
    User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
"""
    
    system_prompt = SYSTEM_PROMPT_FOOD.format(user_profile=user_profile_text,travel_history=travel_history_text,last_results=last_results or "No previous results") + context_text
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    
    return {"messages": [AIMessage(content=response.content)],
             "last_results": [{"handler": Intent.INTENT_D_FOOD_DIETARY.value, "response": response.content}]} 


def handle_friends_based(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Friend-based recommendations"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    time_context = state.get("time_context")
    preferences = state.get("preferences")
    last_results = state.get("last_results") 
    
    context_text = f"""
    Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
    Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
    User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
"""
    
    system_prompt = SYSTEM_PROMPT_FRIENDS_BASED.format(user_profile=user_profile_text,travel_history=travel_history_text,last_results=last_results or "No previous results") + context_text
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    
    return {"messages": [AIMessage(content=response.content)],
             "last_results": [{"handler": Intent.INTENT_E_FRIENDS_BASED.value, "response": response.content}]} 


def handle_safety_practical(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """Safety & practical travel help"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    time_context = state.get("time_context")
    preferences = state.get("preferences")
    last_results = state.get("last_results") 
    
    context_text = f"""
    Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
    Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
    User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
"""
    
    system_prompt = SYSTEM_PROMPT_SAFETY.format(user_profile=user_profile_text,travel_history=travel_history_text,last_results=last_results or "No previous results") + context_text
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    
    return {"messages": [AIMessage(content=response.content)],
             "last_results": [{"handler": Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP.value, "response": response.content}]} 


def handle_fallback(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """General chat fallback"""
    user_id = config["configurable"].get("user_id")
    user_profile_text = get_user_profile_text(store, user_id)

    travel_history_text = get_travel_history_text(store, user_id)

    # context for system prompt
    location = state.get("location")
    time_context = state.get("time_context")
    preferences = state.get("preferences")
    last_results = state.get("last_results") 
    
    context_text = f"""
    Current Location: {location.get('city') if location else 'Unknown'} {f"(lat: {location.get('lat')}, lng: {location.get('lng')})" if location else ''}
    Current Time: {time_context.get('day_of_week')} {time_context.get('local_time')}
    User Preferences: vibe={preferences.get('vibe') if preferences else None}, cuisine={preferences.get('cuisine') if preferences else None}, budget={preferences.get('budget') if preferences else None}
"""
    
    system_prompt = SYSTEM_PROMPT_FALLBACK.format(user_profile=user_profile_text,travel_history=travel_history_text,last_results=last_results or "No previous results") + context_text
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    
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
        print(f"DEBUG location history saved: {history}")

    #save travel history  ---
    travel_namespace = ("travel_history", user_id)
    existing_travel = store.get(travel_namespace, "history")

    if not existing_travel:
        store.put(travel_namespace, "history", DUMMY_TRAVEL_HISTORY)
        logger.info(f"✓ Seeded travel history for {user_id}")
    
    logger.info(f"✓ Saved user profile for {user_id}")
    
    return state

def _build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("context_builder", context_builder)
    builder.add_node("router", router_node)
    builder.add_node(Intent.INTENT_A_NEARBY_GENERIC.value, handle_nearby_generic)
    builder.add_node(Intent.INTENT_B_NEARBY_BY_NEED.value, handle_nearby_by_need)
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
            Intent.INTENT_C_ITINERARY.value: Intent.INTENT_C_ITINERARY.value,
            Intent.INTENT_D_FOOD_DIETARY.value: Intent.INTENT_D_FOOD_DIETARY.value,
            Intent.INTENT_E_FRIENDS_BASED.value: Intent.INTENT_E_FRIENDS_BASED.value,
            Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP.value: Intent.INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP.value,
            "clarification": "clarification",
            "General Chat/ Fallback": "General Chat/ Fallback",
            "Health Emergency": "Health Emergency",
        },
    )
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

    return graph
            # mermaid = graph.get_graph().draw_mermaid()
            # with open("chatbot_graph_4.mmd", "w", encoding="utf-8") as f:
            #     f.write(mermaid)
            # print("Wrote: chatbot_graph_4.mmd")

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
) -> str:

    graph = _get_graph()

    if thread_id is None:
        thread_id = f"travel-{user_id}"
    config = {
        "configurable": {
            "user_id": user_id,
            "thread_id": thread_id,
            "location": location,
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

            