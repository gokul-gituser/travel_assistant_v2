
def format_retrieved_context(
    personal_memories: list,
    conversation_memories: list,
    max_chars: int = 1500,
) -> tuple[str, str]:
    if not personal_memories:
        personal_str = "None"
    else:
        joined = "\n".join(f"- {m}" for m in personal_memories)
        personal_str = joined[:max_chars] + "\n[... truncated]" if len(joined) > max_chars else joined

    if not conversation_memories:
        conversation_str = "None"
    else:
        joined = "\n".join(f"- {m}" for m in conversation_memories)
        conversation_str = joined[:max_chars] + "\n[... truncated]" if len(joined) > max_chars else joined

    return personal_str, conversation_str