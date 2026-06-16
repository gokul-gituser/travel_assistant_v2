from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

extractor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,
    max_completion_tokens=2000  #
    )

async def extract_overpass_tag(message: str) -> tuple[str, str] | None:
    """Use LLM to extract the best Overpass tag from a user message."""
    prompt = f"""You are an OpenStreetMap tag extractor.
        
    Given a user's message, return the single best Overpass API tag to find what they want.

    User message: "{message}"

    Return ONLY a raw JSON object, no markdown, no explanation like:
    {{"tag_key": "amenity", "tag_value": "restaurant"}}

    If the message has no specific place request (e.g. "hello", "how are you"), return:
    {{"tag_key": null, "tag_value": null}}

    Common mappings to guide you:
    - food, eat, hungry, burger, pizza, sushi, lunch, dinner → amenity: restaurant
    - coffee, cafe, tea, latte → amenity: cafe
    - bar, beer, drinks, nightlife, pub → amenity: bar
    - bank, cash, ATM, withdraw, money → amenity: bank
    - doctor, sick, hospital, emergency, hurt → amenity: hospital
    - pharmacy, medicine, prescription, drugs → amenity: pharmacy
    - park, outdoor, picnic, walk, nature → leisure: park
    - gym, workout, fitness, exercise → leisure: fitness_centre
    - hotel, stay, sleep, accommodation → tourism: hotel
    - museum, art, exhibition, gallery → tourism: museum
    - supermarket, groceries, shopping, food store → shop: supermarket
    - fun, entertainment, things to do, activities, explore → leisure: park
    - bowling, mini golf, arcade, games → leisure: amusement_arcade
    - art, gallery, exhibition → tourism: museum
    - music, concert, show, theatre → amenity: theatre
    - market, shopping → shop: mall
    - sightseeing, tourist, landmark, famous → tourism: attraction
    """
    try:
        response = extractor_llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(response.content)
        if data.get("tag_key") and data.get("tag_value"):
            return (data["tag_key"], data["tag_value"])
        return None
    except Exception as e:
        print(f"⚠️ Tag extraction failed: {e}")
        return None


# ---------- Distance ----------

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2 # latitude component
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2 # longitude component
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------- Overpass ----------

def search_overpass(lat, lng, tag_key, tag_value, radius=10000):
    """Category-specific Overpass query."""
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["{tag_key}"="{tag_value}"](around:{radius},{lat},{lng});
      way["{tag_key}"="{tag_value}"](around:{radius},{lat},{lng});
    );
    out center;
    """
    response = requests.post(
        overpass_url,
        data={"data": query}, 
        headers={"User-Agent": "travel-assistant-app"},
        timeout=60
    )
    print("Overpass status:", response.status_code)
    print("Overpass response (first 500 chars):", response.text[:500])
    if response.status_code != 200:
        return []
    try:
        data = response.json()
    except Exception:
        return []

    places = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        plat = el.get("lat") or el["center"]["lat"]
        plng = el.get("lon") or el["center"]["lon"]
        places.append({
            "name": name,
            "type": tags.get(tag_key, tag_value),
            "distance": round(calculate_distance(lat, lng, plat, plng), 1),
        })

    places.sort(key=lambda x: x["distance"])
    return places[:20]

PLACES_TOKEN = "<<<PLACES_LIST>>>"

def build_context_and_display(places: list) -> tuple[str, str]:
    if not places:
        return "", ""

    def dist_label(d):
        return f"{int(d)}m" if d < 1000 else f"{d / 1000:.1f}km"

    context_lines = [
        "NEARBY PLACES — sorted by distance, closest first.",
        "",
        "⚠️ STRICT OUTPUT RULES:",
        "If the user is doing a general search without specific filters:",
        "  1. Write a short intro sentence.",
        "  2. Do NOT list individual places.",
        "  3. Output this token on its own line: <<<PLACES_LIST>>>",
        "",
        "If the user is applying specific filters (like distance, budget, or picking from a list):",
        "  1. DO NOT use the <<<PLACES_LIST>>> token.",
        "  2. Manually list the filtered places yourself.",
        "",
        "PLACE DATA (for your awareness only):",
    ]
    for i, p in enumerate(places, start=1):
        context_lines.append(
            f"  #{i:>2} | {dist_label(p['distance']):<6} | {p['name']} ({p['type']})"
        )
    llm_context = "\n".join(context_lines)

    display_lines = [
        f"{i}. **{p['name']}** ({p['type']}) — {dist_label(p['distance'])}"
        for i, p in enumerate(places, start=1)
    ]
    display_list = "\n".join(display_lines)

    return llm_context, display_list


# ---------- Inject ----------

def inject_places(llm_response: str, display_list: str) -> str:
    if not display_list:
        return llm_response
    if PLACES_TOKEN in llm_response:
        return llm_response.replace(PLACES_TOKEN, f"\n{display_list}")
        
    # If the token is missing, assume the LLM manually formatted 
    # the list based on user filters. Do not blindly append the unfiltered list.
    return llm_response
