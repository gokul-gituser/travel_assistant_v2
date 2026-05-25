"""
FAISS Retrieval Evaluation Script
===================================
Tests recall@k for the two active handlers: nearby_generic and fallback.

HOW TO RUN:
    python eval_faiss_retrieval.py

WHAT IT DOES:
1. Seeds FAISS with realistic past conversations
2. Runs test queries for each handler
3. Checks if expected keywords appear in top-k results
4. Prints recall@k summary and flags weak spots

TUNING:
    Adjust CHUNK_SIZE, OVERLAP, TOP_K at the top to test different configs.
"""

import sys
import os

# ── Point to your travel_assistant package ──────────────────────────────────
# Adjust this path if your package lives elsewhere
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
USER_ID    = "eval_user_001"   # isolated test user so real data isn't polluted

# ── Seed data: realistic past conversations ──────────────────────────────────
# Format: (user_message, assistant_reply, handler_tag)
SEED_CONVERSATIONS = [

    # --- nearby_generic seeds ---
    (
        "Find me a good cafe near me",
        "Here are some cafes near you: Brew & Bean (0.3km), The Coffee House (0.6km), Morning Blend (1.1km).",
        "nearby_generic",
    ),
    (
        "I want a restaurant for dinner tonight",
        "I found several restaurants nearby: Bella Italia (0.5km), The Grill House (0.8km), Spice Garden (1.2km).",
        "nearby_generic",
    ),
    (
        "Show me parks close to my location",
        "There are a few parks nearby: Victoria Park (0.4km), Riverside Gardens (0.9km), Green Meadows (1.5km).",
        "nearby_generic",
    ),
    (
        "Any pharmacies open now near me?",
        "I found two open pharmacies: Boots Pharmacy (0.2km, open until 10pm), LloydsPharmacy (0.7km, open until 9pm).",
        "nearby_generic",
    ),
    (
        "I need a hospital or clinic urgently",
        "The nearest hospital is City General Hospital (1.1km). There's also MediCare Clinic (0.5km) which is open now.",
        "nearby_generic",
    ),
    (
        "Where can I get pizza around here?",
        "Pizza options near you: Napoli Express (0.3km), Pizza Palace (0.6km), Slice & Dice (1.0km).",
        "nearby_generic",
    ),

    # --- fallback seeds ---
    (
        "What's the best time to visit Tokyo?",
        "The best time to visit Tokyo is spring (March to May) for cherry blossoms, or autumn (September to November) for mild weather and fall foliage.",
        "fallback",
    ),
    (
        "I hate crowded touristy places",
        "Noted! I'll keep that in mind and suggest quieter, off-the-beaten-path spots for you.",
        "fallback",
    ),
    (
        "I'm travelling with my elderly mum, she has mobility issues",
        "I'll make sure to suggest wheelchair-accessible venues and avoid places with lots of stairs.",
        "fallback",
    ),
    (
        "Do I need a visa to visit Japan from the UK?",
        "UK citizens can visit Japan visa-free for up to 90 days for tourism purposes.",
        "fallback",
    ),
    (
        "I prefer vegetarian food",
        "Got it, I'll prioritise vegetarian-friendly restaurants and cafes in my recommendations.",
        "fallback",
    ),
    (
        "What currency should I bring to Thailand?",
        "Thailand uses the Thai Baht (THB). It's best to exchange money at local banks or authorised exchange booths for better rates.",
        "fallback",
    ),
]

# ── Test queries with expected keywords ──────────────────────────────────────
# Each entry: (query, [keywords that should appear in at least one top-k result], handler_tag)
TEST_CASES = [

    # nearby_generic queries
    ("coffee shop nearby",          ["cafe", "Coffee", "Brew"],          "nearby_generic"),
    ("places to eat near me",       ["restaurant", "Grill", "Italia"],   "nearby_generic"),
    ("outdoor spaces close to me",  ["park", "Gardens", "Meadows"],      "nearby_generic"),
    ("pharmacy open right now",     ["pharmacy", "Boots", "Lloyds"],     "nearby_generic"),
    ("I need medical help nearby",  ["hospital", "clinic", "MediCare"],  "nearby_generic"),
    ("pizza place near here",       ["pizza", "Napoli", "Pizza"],        "nearby_generic"),

    # fallback queries
    ("when should I go to Japan",         ["Tokyo", "spring", "cherry"],    "fallback"),
    ("I don't like busy tourist areas",   ["crowded", "touristy", "quiet"], "fallback"),
    ("travelling with someone disabled",  ["mobility", "wheelchair", "accessible"], "fallback"),
    ("visa requirements UK",              ["visa", "Japan", "UK"],          "fallback"),
    ("vegetarian options",                ["vegetarian", "restaurants"],    "fallback"),
    ("money in Southeast Asia",           ["Baht", "Thailand", "currency"], "fallback"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def seed_faiss():
    """Add all seed conversations into FAISS under the eval user."""
    texts     = []
    metadatas = []

    for user_msg, assistant_reply, handler in SEED_CONVERSATIONS:
        now = __import__("datetime").datetime.utcnow()
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


def run_eval():
    """Run all test queries and measure recall@k."""
    results_by_handler = {}
    all_pass = 0
    all_fail = 0
    failures = []

    for query, expected_keywords, handler in TEST_CASES:
        hits = search_documents(
            query=query,
            top_k=TOP_K,
            filters={"username": USER_ID},
        )

        retrieved_texts = [h["text"] for h in hits]
        combined = " ".join(retrieved_texts).lower()

        matched_keywords = [kw for kw in expected_keywords if kw.lower() in combined]
        passed = len(matched_keywords) > 0

        if handler not in results_by_handler:
            results_by_handler[handler] = {"pass": 0, "fail": 0}

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
                "got":      retrieved_texts[:2],  # show top 2 for debugging
            })

    return results_by_handler, all_pass, all_fail, failures


def print_report(results_by_handler, all_pass, all_fail, failures):
    total = all_pass + all_fail
    recall = all_pass / total * 100 if total else 0

    print("=" * 55)
    print(f"  FAISS RETRIEVAL EVAL  |  top_k={TOP_K}  chunk={CHUNK_SIZE}  overlap={OVERLAP}")
    print("=" * 55)

    for handler, counts in results_by_handler.items():
        h_total  = counts["pass"] + counts["fail"]
        h_recall = counts["pass"] / h_total * 100 if h_total else 0
        status   = "✅" if h_recall >= 80 else "⚠️ " if h_recall >= 50 else "❌"
        print(f"  {status}  {handler:<20}  {counts['pass']}/{h_total}  ({h_recall:.0f}%)")

    print("-" * 55)
    print(f"  OVERALL recall@{TOP_K}: {all_pass}/{total}  ({recall:.0f}%)")
    print("=" * 55)

    if failures:
        print("\n📋 FAILURES — queries that returned no expected keywords:\n")
        for f in failures:
            print(f"  Handler : {f['handler']}")
            print(f"  Query   : {f['query']}")
            print(f"  Expected: {f['expected']}")
            print(f"  Got     : {f['got'][:1]}")  # first result only
            print()

    # Tuning advice
    print("💡 TUNING ADVICE")
    if recall < 50:
        print("  → recall is low. Try reducing chunk_size (e.g. 150) so chunks are more focused.")
        print("  → Or increase top_k to 8-10 to cast a wider net.")
    elif recall < 80:
        print("  → Decent but not great. Try increasing overlap (e.g. 75) to avoid splitting key phrases across chunks.")
    else:
        print("  → recall looks good. Current chunk_size/overlap/top_k settings are working well.")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔍 Starting FAISS retrieval evaluation...\n")

    seed_faiss()

    print(f"📦 Total vectors in index: {index.ntotal}")
    print(f"📄 Total documents tracked: {len(documents)}\n")

    results_by_handler, all_pass, all_fail, failures = run_eval()

    print_report(results_by_handler, all_pass, all_fail, failures)