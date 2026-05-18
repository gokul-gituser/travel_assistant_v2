from travel_assistant.mem0_store import get_mem0


def add_semantic_memory(
    user_id: str,
    memory: str,
    metadata: dict | None = None,
):
    if not user_id or not memory:
        return

    m = get_mem0()

    try:
        m.add(
            memory,
            user_id=user_id,
            metadata=metadata or {},
        )

    except Exception as e:
        print(f"[MEM0 ADD ERROR] {e}")