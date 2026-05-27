"""
FAISS Retrieval Evaluation Script
===================================
Tests recall@k and precision (irrelevant retrieval tracking) for all handlers.

HOW TO RUN:
    python eval_faiss_retrieval.py

TUNING:
    Adjust CHUNK_SIZE, OVERLAP, TOP_K at the top to test different configs.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from travel_assistant.faiss_store import (
    add_documents,
    search_documents,
    index,
    documents,
    metadata_store,
)

# ── Tunable parameters ───────────────────────────────────────────────────────
CHUNK_SIZE = 300
OVERLAP    = 50
TOP_K      = 5
SCORE_THRESHOLD  = 1.5
USER_ID    = "eval_user_001"

# ── Seed data ────────────────────────────────────────────────────────────────
SEED_CONVERSATIONS = [

    # --- nearby_generic ---
    ("Find me a good cafe near me", "Here are some cafes near you: Brew & Bean (0.3km), The Coffee House (0.6km), Morning Blend (1.1km).", "nearby_generic"),
    ("I want a restaurant for dinner tonight", "I found several restaurants nearby: Bella Italia (0.5km), The Grill House (0.8km), Spice Garden (1.2km).", "nearby_generic"),
    ("Show me parks close to my location", "There are a few parks nearby: Victoria Park (0.4km), Riverside Gardens (0.9km), Green Meadows (1.5km).", "nearby_generic"),
    ("Any pharmacies open now near me?", "I found two open pharmacies: Boots Pharmacy (0.2km, open until 10pm), LloydsPharmacy (0.7km, open until 9pm).", "nearby_generic"),
    ("I need a hospital or clinic urgently", "The nearest hospital is City General Hospital (1.1km). There's also MediCare Clinic (0.5km) which is open now.", "nearby_generic"),
    ("Where can I get pizza around here?", "Pizza options near you: Napoli Express (0.3km), Pizza Palace (0.6km), Slice & Dice (1.0km).", "nearby_generic"),

    # --- fallback ---
    ("What's the best time to visit Tokyo?", "The best time to visit Tokyo is spring (March to May) for cherry blossoms, or autumn (September to November) for mild weather and fall foliage.", "fallback"),
    ("I hate crowded touristy places", "Noted! I'll keep that in mind and suggest quieter, off-the-beaten-path spots for you.", "fallback"),
    ("I'm travelling with my elderly mum, she has mobility issues", "I'll make sure to suggest wheelchair-accessible venues and avoid places with lots of stairs.", "fallback"),
    ("Do I need a visa to visit Japan from the UK?", "UK citizens can visit Japan visa-free for up to 90 days for tourism purposes.", "fallback"),
    ("I prefer vegetarian food", "Got it, I'll prioritise vegetarian-friendly restaurants and cafes in my recommendations.", "fallback"),
    ("What currency should I bring to Thailand?", "Thailand uses the Thai Baht (THB). It's best to exchange money at local banks or authorised exchange booths for better rates.", "fallback"),

    # --- food_dietary ---
    ("I am allergic to nuts, what should I avoid?", "You should avoid dishes containing peanuts, almonds, cashews, and tree nuts. Always inform the restaurant staff about your nut allergy before ordering.", "food_dietary"),
    ("I'm vegan, what can I eat in Japan?", "Japan has growing vegan options. Look for shojin ryori (Buddhist temple food), tofu dishes, and dedicated vegan restaurants in cities like Tokyo and Kyoto.", "food_dietary"),
    ("What's a good high protein meal for travellers?", "Good high protein options include grilled chicken, eggs, legumes, and Greek yogurt. Many local markets also sell boiled eggs and protein-rich street food.", "food_dietary"),
    ("I'm diabetic, what foods should I watch out for?", "Avoid high sugar foods like white rice in large quantities, sugary drinks, and pastries. Opt for whole grains, vegetables, and lean proteins.", "food_dietary"),

    # --- safety_practical ---
    ("Is it safe to travel alone at night in Bangkok?", "Bangkok is generally safe for solo travellers at night in tourist areas like Silom and Sukhumvit. Avoid poorly lit side streets and always use reputable transport apps.", "safety_practical"),
    ("What should I do if I lose my passport abroad?", "Contact your country's nearest embassy or consulate immediately. They can issue an emergency travel document. Also file a police report for insurance purposes.", "safety_practical"),
    ("Are there any health risks in Southeast Asia I should know about?", "Common health risks include dengue fever, food-borne illness, and heat exhaustion. Ensure vaccinations for Hepatitis A, Typhoid, and consider malaria prophylaxis for rural areas.", "safety_practical"),
    ("What travel insurance should I get?", "Look for a policy that covers medical evacuation, trip cancellation, lost luggage, and 24/7 emergency assistance. World Nomads and Allianz are popular choices for travellers.", "safety_practical"),

    # --- friends_based ---
    ("Where did my friend go last weekend?", "Based on your friend's recent activity, they visited Riverside Cafe and checked in at Central Park last weekend.", "friends_based"),
    ("Has any of my friends been to Bali recently?", "Yes, your friend Sarah posted about her trip to Bali last month. She visited Ubud and Seminyak and highly recommended the rice terraces.", "friends_based"),

    # --- itinerary ---
    ("Plan me a 3 day trip to Rome", "Here's a 3-day Rome itinerary: Day 1 — Colosseum, Roman Forum, Palatine Hill. Day 2 — Vatican Museums, St Peter's Basilica, Castel Sant'Angelo. Day 3 — Trastevere, Campo de' Fiori, Piazza Navona.", "itinerary"),
    ("I want a relaxed 2 day trip to Kyoto", "Day 1 — Arashiyama bamboo grove, Tenryu-ji temple, afternoon stroll in Gion. Day 2 — Fushimi Inari shrine in the morning, Nishiki Market, Kinkaku-ji (Golden Pavilion).", "itinerary"),
    ("Plan a family trip to Paris for 4 days", "Day 1 — Eiffel Tower, Champ de Mars picnic. Day 2 — Louvre Museum, Tuileries Garden. Day 3 — Disneyland Paris. Day 4 — Montmartre, Sacre-Coeur, souvenir shopping.", "itinerary"),

    # --- nearby_by_need ---
    ("I'm bored, what can I do nearby?", "Since you're looking for something to do, here are some options near you: City Museum (0.8km) for exhibits, Sunset Bowling Alley (1.2km) for fun, and Riverside Park (0.5km) for a walk.", "nearby_by_need"),
    ("It started raining and I don't have an umbrella, what can I do nearby?", "Since it's raining, here are some indoor options nearby: Westfield Mall (0.6km), City Library (0.9km), and Cineplex Cinema (1.1km) to kill time.", "nearby_by_need"),
    ("I have two kids aged 8 and 10, what activities can we do nearby?", "Great family options nearby: KidZone Play Centre (0.7km), Aquarium of the City (1.3km), and Adventure Mini Golf (1.0km) — all kid-friendly.", "nearby_by_need"),
    ("It's a nice evening, what's a good romantic spot nearby?", "For a romantic evening, try Harbour View Restaurant (0.9km) with waterfront dining, or Moonlight Rooftop Bar (1.2km) for drinks with a view.", "nearby_by_need"),
    ("I need to kill an hour, what's close by?", "To fill an hour nearby: City Coffee Co (0.3km) for a relaxed sit-down, or browse the weekend market at Town Square (0.5km).", "nearby_by_need"),
]

# ── Test cases ───────────────────────────────────────────────────────────────
# Each entry: (query, relevant_keywords, handler)
# relevant_keywords = words that SHOULD appear in top-k results
# A chunk is "relevant" if it contains at least one of these keywords
# A chunk is "irrelevant" if it contains none of them
TEST_CASES = [

    # nearby_generic
    ("coffee shop nearby",          ["cafe", "Coffee", "Brew"],           "nearby_generic"),
    ("places to eat near me",       ["restaurant", "pizza", "Napoli"],    "nearby_generic"),
    ("outdoor spaces close to me",  ["park", "Gardens", "Meadows"],       "nearby_generic"),
    ("pharmacy open right now",     ["pharmacy", "Boots", "Lloyds"],      "nearby_generic"),
    ("I need medical help nearby",  ["hospital", "clinic", "MediCare"],   "nearby_generic"),
    ("pizza place near here",       ["pizza", "Napoli", "Pizza"],         "nearby_generic"),

    # fallback
    ("when should I go to Japan",         ["Tokyo", "spring", "cherry"],              "fallback"),
    ("I don't like busy tourist areas",   ["crowded", "touristy", "quiet"],           "fallback"),
    ("travelling with someone disabled",  ["mobility", "wheelchair", "accessible"],   "fallback"),
    ("visa requirements UK",              ["visa", "Japan", "UK"],                    "fallback"),
    ("vegetarian options",                ["vegetarian", "restaurants"],              "fallback"),
    ("money in Southeast Asia",           ["Baht", "Thailand", "currency"],           "fallback"),

    # food_dietary
    ("I have a nut allergy",              ["nut", "peanut", "allergy"],               "food_dietary"),
    ("vegan food options in Japan",       ["vegan", "tofu", "shojin"],                "food_dietary"),
    ("high protein food while travelling",["protein", "chicken", "eggs"],             "food_dietary"),
    ("food for diabetics",                ["diabetic", "sugar", "glucose"],           "food_dietary"),

    # safety_practical
    ("is it safe to walk alone at night", ["safe", "night", "Bangkok"],              "safety_practical"),
    ("lost my passport what do I do",     ["passport", "embassy", "consulate"],      "safety_practical"),
    ("health risks when travelling Asia", ["dengue", "vaccination", "malaria"],      "safety_practical"),
    ("what travel insurance do I need",   ["insurance", "medical", "evacuation"],    "safety_practical"),

    # friends_based
    ("where did my friend visit recently",["friend", "Riverside", "weekend"],        "friends_based"),
    ("has anyone been to Bali",           ["Bali", "Sarah", "Ubud"],                 "friends_based"),

    # itinerary
    ("plan a trip to Rome",               ["Rome", "Colosseum", "Vatican"],          "itinerary"),
    ("2 day itinerary for Kyoto",         ["Kyoto", "Arashiyama", "Fushimi"],        "itinerary"),
    ("family trip to Paris",              ["Paris", "Eiffel", "Louvre"],             "itinerary"),

    # nearby_by_need
    ("I'm bored what can I do",           ["bored", "Museum", "Bowling"],            "nearby_by_need"),
    ("stuck inside because of rain",      ["raining", "indoor", "Mall"],             "nearby_by_need"),
    ("activities for kids nearby",        ["kids", "KidZone", "Aquarium"],           "nearby_by_need"),
    ("romantic place for tonight",        ["romantic", "Harbour", "Rooftop"],        "nearby_by_need"),
    ("killing time nearby",               ["hour", "Coffee", "market"],              "nearby_by_need"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def seed_faiss():
    texts     = []
    metadatas = []

    for user_msg, assistant_reply, handler in SEED_CONVERSATIONS:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        base = {
            "username": USER_ID,
            "source":   "chat",
            "handler":  handler,
            "year":     now.year,
            "month":    now.month,
            "day":      now.day,
        }
        texts.append(user_msg)
        metadatas.append({**base, "type": "user_message"})
        texts.append(assistant_reply)
        metadatas.append({**base, "type": "assistant_reply"})

    add_documents(texts=texts, metadatas=metadatas)
    print(f"✅ Seeded {len(texts)} chunks ({len(SEED_CONVERSATIONS)} conversations)\n")


def is_relevant(text: str, keywords: list[str]) -> bool:
    """A chunk is relevant if it contains at least one expected keyword."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def run_eval():
    results_by_handler = {}
    all_pass  = 0
    all_fail  = 0
    failures  = []

    for query, expected_keywords, handler in TEST_CASES:
        hits = search_documents(
            query=query,
            top_k=TOP_K,
            filters={"username": USER_ID},
            score_threshold=SCORE_THRESHOLD
        )

        retrieved_texts = [h["text"] for h in hits]
        combined        = " ".join(retrieved_texts).lower()

        # ── Recall: did at least one relevant chunk come back? ──────────────
        matched_keywords = [kw for kw in expected_keywords if kw.lower() in combined]
        passed = len(matched_keywords) > 0

        # ── Precision: how many of the returned chunks are relevant? ────────
        relevant_count   = sum(1 for t in retrieved_texts if is_relevant(t, expected_keywords))
        irrelevant_count = len(retrieved_texts) - relevant_count

        if handler not in results_by_handler:
            results_by_handler[handler] = {
                "pass": 0, "fail": 0,
                "total_retrieved": 0,
                "total_relevant":  0,
                "total_irrelevant": 0,
                "noisy_queries": [],   # queries where irrelevant chunks appeared
            }

        results_by_handler[handler]["total_retrieved"]  += len(retrieved_texts)
        results_by_handler[handler]["total_relevant"]   += relevant_count
        results_by_handler[handler]["total_irrelevant"] += irrelevant_count

        if irrelevant_count > 0:
            results_by_handler[handler]["noisy_queries"].append({
                "query":      query,
                "irrelevant": irrelevant_count,
                "relevant":   relevant_count,
                "chunks":     retrieved_texts,
            })

        if passed:
            results_by_handler[handler]["pass"] += 1
            all_pass += 1
        else:
            results_by_handler[handler]["fail"] += 1
            all_fail += 1
            failures.append({
                "handler":  handler,
                "query":    query,
                "expected": expected_keywords,
                "got":      retrieved_texts[:2],
            })

    return results_by_handler, all_pass, all_fail, failures


def print_report(results_by_handler, all_pass, all_fail, failures):
    total  = all_pass + all_fail
    recall = all_pass / total * 100 if total else 0

    total_retrieved_all  = sum(h["total_retrieved"]  for h in results_by_handler.values())
    total_relevant_all   = sum(h["total_relevant"]   for h in results_by_handler.values())
    total_irrelevant_all = sum(h["total_irrelevant"] for h in results_by_handler.values())
    overall_precision    = total_relevant_all / total_retrieved_all * 100 if total_retrieved_all else 0

    print("=" * 70)
    print(f"  FAISS RETRIEVAL EVAL  |  top_k={TOP_K}  chunk={CHUNK_SIZE}  overlap={OVERLAP}")
    print("=" * 70)
    print(f"  {'Handler':<22}  {'Recall':>8}  {'Precision':>10}  {'Noise/query':>12}")
    print("-" * 70)

    for handler, counts in results_by_handler.items():
        h_total     = counts["pass"] + counts["fail"]
        h_recall    = counts["pass"] / h_total * 100 if h_total else 0
        h_precision = counts["total_relevant"] / counts["total_retrieved"] * 100 if counts["total_retrieved"] else 0
        h_noise     = counts["total_irrelevant"] / h_total if h_total else 0
        r_status    = "✅" if h_recall >= 80    else "⚠️ " if h_recall >= 50    else "❌"
        p_status    = "✅" if h_precision >= 60 else "⚠️ " if h_precision >= 40 else "❌"
        print(
            f"  {r_status} {handler:<22}  "
            f"{counts['pass']}/{h_total} ({h_recall:.0f}%)  "
            f"{p_status} {h_precision:.0f}%  "
            f"  {h_noise:.1f} irrelevant"
        )

    print("-" * 70)
    print(f"  OVERALL recall@{TOP_K}:  {all_pass}/{total} ({recall:.0f}%)")
    print(f"  OVERALL precision:    {total_relevant_all}/{total_retrieved_all} ({overall_precision:.0f}%)")
    print(f"  OVERALL noise:        {total_irrelevant_all} irrelevant chunks across all queries")
    print("=" * 70)

    # ── Noisy queries detail ─────────────────────────────────────────────────
    noisy_found = False
    for handler, counts in results_by_handler.items():
        if counts["noisy_queries"]:
            if not noisy_found:
                print("\n📋 IRRELEVANT RETRIEVALS — queries where noise chunks appeared:\n")
                noisy_found = True
            for nq in counts["noisy_queries"]:
                print(f"  Handler : {handler}")
                print(f"  Query   : {nq['query']}")
                print(f"  Relevant: {nq['relevant']}  Irrelevant: {nq['irrelevant']}")
                print(f"  Chunks  :")
                for i, chunk in enumerate(nq["chunks"]):
                    print(f"    [{i+1}] {chunk[:100]}")
                print()

    if not noisy_found:
        print("\n✅ No irrelevant chunks detected in any query.\n")

    # ── Recall failures ──────────────────────────────────────────────────────
    if failures:
        print("📋 RECALL FAILURES — queries that returned no expected keywords:\n")
        for f in failures:
            print(f"  Handler : {f['handler']}")
            print(f"  Query   : {f['query']}")
            print(f"  Expected: {f['expected']}")
            print(f"  Got     : {f['got'][:1]}")
            print()

    # ── Tuning advice ────────────────────────────────────────────────────────
    print("💡 TUNING ADVICE")
    if recall < 50:
        print("  → Recall is low. Try reducing chunk_size (e.g. 150) so chunks are more focused.")
        print("  → Or increase top_k to 8-10 to cast a wider net.")
    elif recall < 80:
        print("  → Decent recall but not great. Try increasing overlap (e.g. 75).")
    else:
        print("  → Recall looks good.")

    if overall_precision < 60:
        print("  → Precision is low — too many irrelevant chunks coming back.")
        print("  → Consider adding a score_threshold to filter weak matches (e.g. score_threshold=1.5).")
        print("  → Or reduce top_k to return fewer but more focused results.")
    elif overall_precision < 80:
        print("  → Precision is moderate. Some noise is normal with semantic search.")
    else:
        print("  → Precision looks good. Noise is within acceptable range.")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔍 Starting FAISS retrieval evaluation...\n")

    seed_faiss()

    print(f"📦 Total vectors in index: {index.ntotal}")
    print(f"📄 Total documents tracked: {len(documents)}\n")

    results_by_handler, all_pass, all_fail, failures = run_eval()

    print_report(results_by_handler, all_pass, all_fail, failures)