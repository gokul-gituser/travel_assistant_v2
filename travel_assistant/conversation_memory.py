from faiss_store import add_documents


def chat_interaction(
    user_id: str,
    user_message: str,
    assistant_reply: str,
):

    texts = [
        user_message,
        assistant_reply,
    ]

    metadatas = [
        {
            "username": user_id,
            "type": "user_message",
            "source": "chat",
        },
        {
            "username": user_id,
            "type": "assistant_reply",
            "source": "chat",
        }
    ]

    add_documents(
        texts=texts,
        metadatas=metadatas,
    )

    print("\n=== SAVING CHAT TO FAISS ===")
    print("USER:", user_message)
    print("ASSISTANT:", assistant_reply)
    print("============================\n")