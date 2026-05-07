from mem0 import Memory
import os

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    }
}

_mem0 = None

def get_mem0():
    global _mem0
    if _mem0 is None:
        _mem0 = Memory.from_config(config)
    return _mem0

class Mem0Store:
    def get(self, namespace: tuple, key: str):
        user_id = namespace[-1]
        ns_str = "_".join(namespace)
        query = f"{ns_str}:{key}"
        results = get_mem0().search(
            query,
            filters={"user_id": user_id},  # ← changed
            limit=1
        )
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
        get_mem0().add(
            content,
            user_id=user_id,  # ← add() still uses user_id directly
            metadata={"namespace": ns_str, "key": key, "value": value}
        )