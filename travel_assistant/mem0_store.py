# mem0_store.py
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    }
}

_mem0 = Memory.from_config(config)

class Mem0Store:
    """Drop-in replacement for RedisStore with the same .get()/.put() interface"""

    def get(self, namespace: tuple, key: str):
        user_id = namespace[-1]
        ns_str = "_".join(namespace)
        query = f"{ns_str}:{key}"
        results = _mem0.search(query, user_id=user_id, limit=1)
        if results and results[0]:
            class _Item:
                def __init__(self, value):
                    self.value = value
            return _Item(results[0]["metadata"].get("value"))
        return None

    def put(self, namespace: tuple, key: str, value):
        user_id = namespace[-1]
        ns_str = "_".join(namespace)
        content = f"{ns_str}:{key} = {value}"
        _mem0.add(content, user_id=user_id, metadata={"namespace": ns_str, "key": key, "value": value})