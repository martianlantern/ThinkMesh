import hashlib
import orjson
from typing import Any, Dict
from diskcache import Cache as DC

class Cache:
    def __init__(self, enabled: bool = True, ttl_s: int = 86400, path: str = ".thinkmesh_cache"):
        self.enabled = enabled
        self.ttl = ttl_s
        self.db = DC(path) if enabled else None

    def key(self, adapter: str, model: str, prompt: str, params: Dict[str, Any]) -> str:
        h = hashlib.sha256(orjson.dumps([adapter, model, prompt, params])).hexdigest()
        return h

    def get(self, k: str):
        if not self.enabled:
            return None
        return self.db.get(k, default=None)

    def set(self, k: str, v: Any):
        if not self.enabled:
            return
        self.db.set(k, v, expire=self.ttl)
