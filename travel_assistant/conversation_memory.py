from .faiss_store import add_documents, save_index  
from datetime import datetime, timezone
import uuid

def chat_interaction(
    user_id: str,
    user_message: str,
    assistant_reply: str,
    handler: str | None = None,

):

    now = datetime.now(timezone.utc)
    base_meta = {
        "username": user_id,
        "source": "chat",
        "handler": handler,
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "conversation_id": str(uuid.uuid4()),       # ← ties user_message + assistant_reply together
        "timestamp": now.isoformat(),               # ← exact time for traceability
    }
    texts = [user_message, assistant_reply]
    metadatas = [
        {**base_meta, "type": "user_message"},
        {**base_meta, "type": "assistant_reply"},
    ]

    add_documents(
        texts=texts,
        metadatas=metadatas,
    )
    save_index()

    print("\n=== SAVING CHAT TO FAISS ===")
    print("USER:", user_message)
    print("ASSISTANT:", assistant_reply)
    print("============================\n")