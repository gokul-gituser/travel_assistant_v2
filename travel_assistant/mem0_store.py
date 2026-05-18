# mem0_store.py
from mem0 import Memory
from functools import lru_cache


config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333}
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini"}
    }
}

@lru_cache(maxsize=1)
def get_mem0() -> Memory:
    return Memory.from_config(CONFIG)


class Mem0Store:
    """
    Temporary compatibility wrapper.
    Existing chatbot imports still expect this class.
    """

    pass