import asyncio
import time
from collections import Counter, OrderedDict
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
        # Per-owner (user) counters; the cache itself stays shared
        self._owner_hits: Counter = Counter()
        self._owner_misses: Counter = Counter()
        self._owners: dict[str, str] = {}  # key -> owner

    async def get(self, key: str, owner: Optional[str] = None) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                self._record(owner, hit=False)
                return None
            
            value, expiry = self._cache[key]
            if time.time() > expiry:
                self._cache.pop(key)
                self._owners.pop(key, None)
                self._record(owner, hit=False)
                return None
            
            # LRU - move accessed item to end
            self._cache.move_to_end(key)
            self._record(owner, hit=True)
            return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None, owner: Optional[str] = None):
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            
            expiry = time.time() + (ttl or self._default_ttl)
            self._cache[key] = (value, expiry)
            if owner is not None:
                self._owners[key] = owner
            
            # Evict least recently used (first item) if over capacity
            if len(self._cache) > self._capacity:
                evicted, _ = self._cache.popitem(last=False)
                self._owners.pop(evicted, None)

    def _record(self, owner: Optional[str], hit: bool):
        if hit:
            self._hits += 1
        else:
            self._misses += 1
        if owner is not None:
            (self._owner_hits if hit else self._owner_misses)[owner] += 1

    async def get_stats(self, owner: Optional[str] = None):
        async with self._lock:
            stats = {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "capacity": self._capacity,
            }
            if owner is not None:
                stats["user"] = {
                    "hits": self._owner_hits[owner],
                    "misses": self._owner_misses[owner],
                    "size": sum(1 for o in self._owners.values() if o == owner),
                }
            return stats

# Global semantic cache representing a fast memory layer
semantic_cache = AsyncLRUCache(capacity=500, default_ttl=86400)
