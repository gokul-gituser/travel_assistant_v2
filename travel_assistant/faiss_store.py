
import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from .chunking import chunk_text
load_dotenv()
import logging

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

dimension = 1536

index = faiss.IndexFlatL2(dimension)

documents = []

metadata_store = []

INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "faiss_metadata.json"

def get_embedding(text: str,retries: int = 3):
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            logger.debug("get_embedding: %.3fs", time.perf_counter() - t0)
            return response.data[0].embedding
        except Exception as e:
            if attempt == retries - 1:
                logger.error("get_embedding failed after %d attempts: %s | text=%.80s", retries, e, text)
                raise
            logger.warning("[EMBEDDING RETRY %d] %s", attempt + 1, e)
            time.sleep(2 ** attempt)  # 1s, 2s, 4s


def get_embeddings(texts: list[str], retries: int = 3):
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            logger.debug("get_embeddings: %d texts in %.3fs", len(texts), time.perf_counter() - t0)

            return [
                item.embedding
                for item in response.data
            ]
        except Exception as e:
            if attempt == retries - 1:
                logger.error("get_embeddings failed after %d attempts: %s | first_text=%.80s", retries, e, texts[0] if texts else "")
                raise
            logger.warning("[EMBEDDING RETRY %d] %s", attempt + 1, e)
            time.sleep(2 ** attempt)  # 1s, 2s, 4s


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

    logger.info("add_document: %d chunk(s) added | total in index: %d", len(chunks), index.ntotal)


def add_documents(
    texts: list[str],
    metadatas: list[dict] | None = None,
):

    if not texts:
        logger.warning("add_documents called with empty texts — skipping")

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

    t0 = time.perf_counter()
    vectors = get_embeddings(all_chunks)

    arr = np.array(vectors, dtype="float32")

    index.add(arr)

    documents.extend(all_chunks)

    metadata_store.extend(all_metadata)

    logger.info(
        "add_documents: %d text(s) → %d chunk(s) | embed=%.3fs | total in index: %d",
        len(texts), len(all_chunks), embed_latency, index.ntotal,
    )


def search_documents(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
    score_threshold: float | None = None,

):

    if not documents:
            logger.warning("search_documents called but index is empty")
            return []
 
    t0 = time.perf_counter()
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
                logger.debug("Chunk idx=%d score=%.4f exceeded threshold=%.4f — skipped", idx, score, score_threshold)
                continue
        
        logger.debug("FAISS match | score=%.4f | text=%.80s", score, documents[idx])


        print("\n===== FAISS MATCH =====")
        print("QUERY:", query)
        print("TEXT:", documents[idx])
        print("METADATA:", metadata)
        print("SCORE:", score)
        print("=======================\n")

        results.append({
            "text": documents[idx],
            "metadata": metadata,
            "score": score,
        })

        seen.add(idx)
    
    logger.info(
        "search_documents: query=%.60s | top_k=%d | returned=%d | latency=%.3fs",
        query, top_k, len(results), time.perf_counter() - t0,
    )

    return results


def save_index():
    t0 = time.perf_counter()


    faiss.write_index(index, INDEX_FILE)

    payload = []

    for i in range(len(documents)):

        payload.append({
            "text": documents[i],
            "metadata": metadata_store[i],
        })

    with open(METADATA_FILE, "w", encoding="utf-8") as f:

        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("save_index: %d documents saved in %.3fs", len(documents), time.perf_counter() - t0)

    print("✅ FAISS index saved")


def load_index():

    global index
    global documents
    global metadata_store

    if not os.path.exists(INDEX_FILE):
        logger.warning("load_index: no index file found at '%s' — starting fresh", INDEX_FILE)

        print("⚠️ No FAISS index file found")
        return

    if not os.path.exists(METADATA_FILE):
        logger.warning("load_index: no metadata file found at '%s' — starting fresh", METADATA_FILE)
        print("⚠️ No metadata file found")
        return

    t0 = time.perf_counter()

    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:

        payload = json.load(f)

    documents = [item["text"] for item in payload]

    metadata_store = [
        item["metadata"]
        for item in payload
    ]

    # Validate index integrity
    if index.ntotal != len(documents):
        logger.error(
            "load_index: integrity check FAILED — vectors=%d documents=%d",
            index.ntotal, len(documents),
        )
    else:
        logger.info(
            "load_index: %d documents loaded in %.3fs | integrity OK",
            len(documents), time.perf_counter() - t0,
        )

    print(f"✅ Loaded {len(documents)} documents")