import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_manager import AsyncLRUCache  # noqa: E402


def test_per_user_stats_and_eviction():
    async def run():
        c = AsyncLRUCache(capacity=2)
        assert await c.get("a:q1", owner="a") is None          # a miss
        await c.set("a:q1", "x", owner="a")
        assert await c.get("a:q1", owner="a") == "x"           # a hit
        assert await c.get("b:q1", owner="b") is None          # b miss
        await c.set("b:q1", "y", owner="b")
        await c.set("b:q2", "z", owner="b")                    # evicts a:q1

        a, b = await c.get_stats("a"), await c.get_stats("b")
        assert (a["hits"], a["misses"], a["size"]) == (1, 2, 2)  # server-wide
        assert a["user"] == {"hits": 1, "misses": 1, "size": 0}
        assert b["user"] == {"hits": 0, "misses": 1, "size": 2}

    asyncio.run(run())


def test_clear_owner_drops_only_that_owner():
    async def run():
        c = AsyncLRUCache(capacity=10)
        await c.set("a:q1", "x", owner="a")
        await c.set("a:q2", "x", owner="a")
        await c.set("b:q1", "y", owner="b")
        assert await c.clear_owner("a") == 2
        assert await c.get("a:q1", owner="a") is None
        assert await c.get("b:q1", owner="b") == "y"
        assert (await c.get_stats("a"))["user"]["size"] == 0
        assert await c.clear_owner("nobody") == 0

    asyncio.run(run())
