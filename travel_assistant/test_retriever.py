# test_retriever.py

from retriever import retrieve_context
from faiss_store import add_documents

add_documents(
    [
        "Rahul visited Kyoto temples in Japan",
        "Rahul likes sushi and anime",
    ],
    [
        {
            "username": "rahul",
            "year": 2025,
        },
        {
            "username": "rahul",
            "year": 2025,
        }
    ]
)

context = retrieve_context(
    user_id="rahul",
    query="Where did Rahul travel?"
)

print(context)