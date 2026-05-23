from faiss_store import (
    add_documents,
    search_documents,
    save_index,
    load_index,
)

texts = [
    """
    Paris was amazing. We visited Eiffel Tower and Louvre Museum.

    Later we discussed Redis caching architecture and FAISS indexing.

    Finally we planned a Tokyo sushi vacation with beach resorts.
    """
]

metadatas = [
    {
        "username": "gokul",
        "year": 2026,
        "source": "travel_post",
    }
]

# Index documents
add_documents(texts, metadatas)
add_documents(
    [
        "Rahul visited Kyoto temples and Japan cherry blossom festival"
    ],
    [
        {
            "username": "rahul",
            "year": 2025,
            "source": "travel_post",
        }
    ]
)

# Persist to disk
save_index()

# Simulate app restart
load_index()

results = search_documents(
    query="japan trip",
    filters={
        "username": "rahul"
    }
)

print("\nRESULTS:\n")

for item in results:
    print(item)