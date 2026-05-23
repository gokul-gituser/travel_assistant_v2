
import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from .chunking import chunk_text
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

dimension = 1536

index = faiss.IndexFlatL2(dimension)

documents = []

metadata_store = []

INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "faiss_metadata.json"

def get_embedding(text: str):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return response.data[0].embedding


def get_embeddings(texts: list[str]):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    return [
        item.embedding
        for item in response.data
    ]


def add_document(

    text: str,
    metadata: dict | None = None,
):

    chunks = chunk_text(
        text,
        chunk_size=300,
        overlap=50,
    )

    for chunk_index, chunk in enumerate(chunks):

        vector = get_embedding(chunk)

        arr = np.array([vector], dtype="float32")

        index.add(arr)

        documents.append(chunk)

        chunk_metadata = {
            **(metadata or {}),
            "chunk_id": chunk_index,
        }

        metadata_store.append(chunk_metadata)


def add_documents(
    texts: list[str],
    metadatas: list[dict] | None = None,
):

    if not texts:
        return

    if metadatas is None:
        metadatas = [{} for _ in texts]

    all_chunks = []
    all_metadata = []

    for text, metadata in zip(texts, metadatas):

        chunks = chunk_text(
            text,
            chunk_size=300,
            overlap=50,
        )

        for chunk_index, chunk in enumerate(chunks):

            all_chunks.append(chunk)

            all_metadata.append({
                **metadata,
                "chunk_id": chunk_index,
            })

    vectors = get_embeddings(all_chunks)

    arr = np.array(vectors, dtype="float32")

    index.add(arr)

    documents.extend(all_chunks)

    metadata_store.extend(all_metadata)


def search_documents(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
    score_threshold: float | None = None,

):

    vector = get_embedding(query)

    arr = np.array([vector], dtype="float32")

    k = min(top_k, len(documents))

    distances, indices = index.search(arr, k)

    results = []

    seen = set()

    for position, idx in enumerate(indices[0]):

        if idx >= len(documents):
            continue

        if idx in seen:
            continue

        metadata = metadata_store[idx]

        # Apply metadata filters
        if filters:

            matched = True

            for key, value in filters.items():

                if metadata.get(key) != value:
                    matched = False
                    break

            if not matched:
                continue

        score = float(distances[0][position])

        # Lower score = better match in L2 distance
        if score_threshold is not None:

            if score > score_threshold:
                continue

        results.append({
            "text": documents[idx],
            "metadata": metadata,
            "score": score,
        })

        seen.add(idx)

    return results


def save_index():

    faiss.write_index(index, INDEX_FILE)

    payload = []

    for i in range(len(documents)):

        payload.append({
            "text": documents[i],
            "metadata": metadata_store[i],
        })

    with open(METADATA_FILE, "w", encoding="utf-8") as f:

        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✅ FAISS index saved")


def load_index():

    global index
    global documents
    global metadata_store

    if not os.path.exists(INDEX_FILE):
        print("⚠️ No FAISS index file found")
        return

    if not os.path.exists(METADATA_FILE):
        print("⚠️ No metadata file found")
        return

    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:

        payload = json.load(f)

    documents = [item["text"] for item in payload]

    metadata_store = [
        item["metadata"]
        for item in payload
    ]

    print(f"✅ Loaded {len(documents)} documents")