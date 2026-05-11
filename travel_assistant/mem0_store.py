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
        ns_str  = "_".join(str(n) for n in namespace)

        all_results = get_mem0().get_all(filters={"user_id": user_id})

        # mem0 returns either a list or {"results": [...]}
        items = all_results if isinstance(all_results, list) else all_results.get("results", [])

        for item in items:
            meta = item.get("metadata", {})
            if meta.get("namespace") == ns_str and meta.get("key") == key:
                class _Item:
                    def __init__(self, value):
                        self.value = value
                return _Item(meta.get("value"))

        return None

    def put(self, namespace: tuple, key: str, value):
        user_id = namespace[-1]
        ns_str  = "_".join(str(n) for n in namespace)
        content = f"{ns_str}:{key} = {value}"
        
        print(f"🟡 mem0 PUT called — user_id={user_id}, ns={ns_str}, key={key}")
        
        try:
            get_mem0().add(
                content,
                user_id=user_id,
                metadata={"namespace": ns_str, "key": key, "value": value}
            )
            print(f"✅ mem0 PUT success — user_id={user_id}, key={key}")
        except Exception as e:
            print(f"❌ mem0 PUT failed — {e}")