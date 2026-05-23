from .mem0_store import get_mem0


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


def add_travel_memory(
    user_id: str,
    from_city: str,
    to_city: str,
    from_country: str | None = None,
    to_country: str | None = None,
    distance_km: float | None = None,
    detected_at: str | None = None,
):
    
    memory_text = f"Travelled from {from_city} to {to_city}"

    if to_country:
        memory_text += f" in {to_country}"

    if distance_km:
        memory_text += f" covering {distance_km:.2f} km"

    add_semantic_memory(
        user_id=user_id,
        memory=memory_text,
        metadata={
            "type": "travel_history",
            "source": "travel_detection",
            "from_city": from_city,
            "to_city": to_city,
            "from_country": from_country,
            "to_country": to_country,
            "distance_km": distance_km,
            "detected_at": detected_at,
        },
    )