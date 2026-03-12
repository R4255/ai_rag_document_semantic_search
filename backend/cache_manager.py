import asyncio
import time
from collections import OrderedDict
from typing import Any, Optional

class AsyncLRUCache:
    """
    An async-safe LRU Cache with TTL support.
    Showcases core eviction algorithms and thread-safe operations.
    Inspired by professional distributed cache implementations.
    """
    def __init__(self, capacity: int, default_ttl: int = 3600):
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            value, expiry = self._cache[key]
            if time.time() > expiry:
                self._cache.pop(key)
                self._misses += 1
                return None
            
            # LRU - move accessed item to end
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            
            expiry = time.time() + (ttl or self._default_ttl)
            self._cache[key] = (value, expiry)
            
            # Evict least recently used (first item) if over capacity
            if len(self._cache) > self._capacity:
                self._cache.popitem(last=False)

    async def get_stats(self):
        async with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "capacity": self._capacity,
            }

# Global semantic cache representing a fast memory layer
semantic_cache = AsyncLRUCache(capacity=500, default_ttl=86400)
