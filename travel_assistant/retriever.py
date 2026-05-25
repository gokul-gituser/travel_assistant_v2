
from typing import Dict, List

from .mem0_store import get_mem0
from .faiss_store import search_documents

logger = logging.getLogger(__name__)

def search_user_memory(
    user_id: str,
    query: str,
    limit: int = 5,
) -> List[str]:

    if not user_id:
        logger.warning("search_user_memory called with empty user_id — skipping")

        return []

    m = get_mem0()

    try:
        results = m.search(
            query=query,
            limit=limit,
            filters={
                "user_id": user_id,
            }
        )
        latency = time.perf_counter() - t0


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


#FAISS retrieval layer


def search_conversations(
    user_id: str,
    query: str,
    limit: int = 5,
) -> List[str]:

    try:

        results = search_documents(
            query=query,
            top_k=limit,
            filters={
                "username": user_id,
            }
        )

        return [
            item["text"]
            for item in results
        ]

    except Exception as e:
        print(f"[FAISS SEARCH ERROR] {e}")
        return []


def retrieve_context(
    user_id: str,
    query: str,
) -> Dict[str, List[str]]:

    personal_memories = search_user_memory(
        user_id=user_id,
        query=query,
        limit=5,
    )

    conversation_memories = search_conversations(
        user_id=user_id,
        query=query,
        limit=5,
    )

    context = {
        "personal_memories": personal_memories,
        "conversation_memories": conversation_memories,
    }

    print("\n===== RETRIEVED CONTEXT =====")
    print(context)
    print("================================\n")

    return context