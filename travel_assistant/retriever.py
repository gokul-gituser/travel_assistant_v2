
from typing import Dict, List

from travel_assistant.mem0_store import get_mem0


def search_user_memory(
    user_id: str,
    query: str,
    limit: int = 5,
) -> List[str]:

    if not user_id:
        return []

    m = get_mem0()

    try:
        results = m.search(
            query=query,
            user_id=user_id,
            limit=limit,
        )

        items = (
            results
            if isinstance(results, list)
            else results.get("results", [])
        )

        memories = []

        for item in items:
            memory = item.get("memory")

            if memory:
                memories.append(memory)

        return memories

    except Exception as e:
        print(f"[MEM0 SEARCH ERROR] {e}")
        return []


# Future FAISS retrieval layer
# Boss's upcoming task plugs in here

def search_conversations(
    user_id: str,
    query: str,
    limit: int = 5,
) -> List[str]:

    return []


# Unified retrieval interface

def retrieve_context(
    user_id: str,
    query: str,
) -> Dict[str, List[str]]:

    return {
        "personal_memories": search_user_memory(
            user_id=user_id,
            query=query,
            limit=5,
        ),
        "conversation_memories": search_conversations(
            user_id=user_id,
            query=query,
            limit=5,
        ),
    }